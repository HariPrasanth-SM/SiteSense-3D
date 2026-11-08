"""Persistent gsplat batch rendering. No masks/embeddings or navigation changes."""
from __future__ import annotations

import csv
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from .common import (SCHEMA, camera_record, file_sha256, frame_complete,
                     load_plan, read_json, save_frame, write_json)


def check_bindings(out, plan):
    out = Path(out)
    scene_path = Path(plan["inputs"]["scene_path"])
    if not scene_path.exists():
        raise FileNotFoundError(f"Original Gaussian PLY not found: {scene_path}")
    import avatar_demo.core as core
    import avatar_demo.render_worker as worker
    binding = {"plan_hash": plan["plan_hash"], "scene_sha256": file_sha256(scene_path),
               "scene_bytes": scene_path.stat().st_size,
               "source_code": {"avatar_core": file_sha256(core.__file__),
                               "avatar_scene_loader": file_sha256(worker.__file__),
                               "capture": file_sha256(__file__),
                               "common": file_sha256(Path(__file__).with_name("common.py"))}}
    path = out / "render_binding.json"
    if path.exists() and read_json(path) != binding:
        raise ValueError("Scene/plan/dependency source changed since capture started. Use a new dataset directory.")
    write_json(path, binding)
    return binding


def render(out, pilot=False, limit=None):
    out = Path(out).resolve()
    plan = load_plan(out)
    cfg = plan["config"]
    views = [v for v in plan["frames"] if v["pilot"] or not pilot]
    # Verify saved frame checksums; incomplete or altered frames are rerendered.
    todo = [v for v in views if not frame_complete(out / "frames" / v["frame_id"], plan["plan_hash"])]
    if limit is not None: todo = todo[:limit]
    print(f"Selected {len(views)} planned views; {len(todo)} require rendering.", flush=True)
    if not todo:
        report(out); return
    # One extra frame and a conservative PNG allowance, plus 0.5 GiB reserve.
    estimated = len(todo) * cfg["width"] * cfg["height"] * 18 + 512 * 1024**2
    free_bytes = shutil.disk_usage(out).free
    if free_bytes < estimated:
        raise RuntimeError(f"Insufficient free disk space. Conservative estimate: {estimated/1024**3:.2f} GiB; "
                           f"available: {free_bytes/1024**3:.2f} GiB.")
    # Acquire a POSIX lock: avoid two captures sharing this dataset / duplicating GPU memory.
    import fcntl
    lock = (out / ".render.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another renderer is already writing this dataset.")
    try:
        binding = check_bindings(out, plan)
        import torch
        from gsplat import rasterization
        from avatar_demo.render_worker import _load_scene
        from avatar_demo.core import camera_matrices
        torch.cuda.init()
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "gsplat": importlib.metadata.version("gsplat"), "gpu": torch.cuda.get_device_name(0),
                "numpy": np.__version__, "schema": SCHEMA, "synthetic_views": True,
                "plan_hash": plan["plan_hash"], "scene_sha256": binding["scene_sha256"]}
        env_file = out / "render_environment.json"
        if env_file.exists():
            old = read_json(env_file)
            for key in ("torch", "torch_cuda", "gsplat", "plan_hash", "scene_sha256"):
                if old[key] != info[key]:
                    raise ValueError(f"Render environment changed ({key}); use a new dataset for reproducibility.")
        else:
            write_json(env_file, info)
        print("Loading original Gaussian PLY onto GPU ONCE. Keep avatar/RViz/FastSAM closed during capture.", flush=True)
        scene = _load_scene(plan["inputs"]["scene_path"], torch, cfg["sh_degree"])
        print(f"Loaded {len(scene[0]):,} Gaussians; rendering one camera at a time.", flush=True)
        started_run = time.perf_counter()
        for i, view in enumerate(todo, 1):
            started = time.perf_counter()
            K, c2w, w2c = camera_matrices(view["x"], view["y"], math.radians(view["yaw_deg"]),
                                         view["camera_height_m"], view["pitch_deg"], scale=cfg["scale"],
                                         floor_y=cfg["floor_y_scene"], width=cfg["width"],
                                         height_px=cfg["height"], fov_y_deg=cfg["fov_y_deg"])
            with torch.inference_mode():
                pixels, alphas, meta = rasterization(
                    means=scene[0], quats=scene[1], scales=scene[2], opacities=scene[3], colors=scene[4],
                    viewmats=torch.from_numpy(w2c[None]).to("cuda"), Ks=torch.from_numpy(K[None]).to("cuda"),
                    width=cfg["width"], height=cfg["height"], sh_degree=cfg["sh_degree"],
                    render_mode="RGB+ED", packed=True,
                    near_plane=cfg["near_m"]/cfg["scale"], far_plane=cfg["far_m"]/cfg["scale"])
                array = pixels[0].detach().cpu().numpy()
                alpha = alphas[0, :, :, 0].detach().cpu().numpy().copy()
                rgb = np.rint(np.clip(array[:, :, :3], 0, 1) * 255).astype(np.uint8)
                raw_depth = array[:, :, 3].astype(np.float32).copy()
                del pixels, alphas, meta, array
            render_seconds = time.perf_counter() - started
            folder = out / "frames" / view["frame_id"]
            save_frame(folder, plan, view, rgb, raw_depth, alpha, render_seconds)
            total = time.perf_counter() - started
            print(f"[{i}/{len(todo)}] {view['frame_id']} | render {render_seconds:.3f}s | "
                  f"including writes {total:.3f}s | x={view['x']:.2f}, y={view['y']:.2f}, "
                  f"yaw={view['yaw_deg']:.0f}, pitch={view['pitch_deg']:.0f}", flush=True)
        print(f"Capture finished; {time.perf_counter()-started_run:.1f}s after scene load.", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Completed frames remain intact. Re-run the same render command to resume.", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    report(out)


def report(out):
    out = Path(out)
    plan = load_plan(out)
    complete, pending, qa_rows = [], [], []
    # Also write a directly joinable index for subsequent masking/embedding stages.
    index = []
    for v in plan["frames"]:
        folder = out / "frames" / v["frame_id"]
        ok = frame_complete(folder, plan["plan_hash"])
        rel = f"frames/{v['frame_id']}"
        index.append({**v, "complete": ok, "rgb_path": f"{rel}/rgb.png",
                      "depth_scene_path": f"{rel}/depth_scene.npy", "depth_m_path": f"{rel}/depth_m.npy",
                      "alpha_path": f"{rel}/alpha.npy", "camera_path": f"{rel}/camera.json"})
        if ok:
            complete.append(v)
            qa = read_json(folder / "qa.json")
            qa_rows.append({"frame_id": v["frame_id"], "station_id": v["station_id"],
                            "pilot": v["pilot"], "unknown_capture_only": v["unknown_capture_only"],
                            "valid_fraction": qa["valid_fraction"], "alpha_mean": qa["alpha_mean"],
                            "render_seconds": qa["render_seconds"], "flags": ";".join(qa["flags"])})
        else:
            pending.append(v["frame_id"])
    with (out / "index.jsonl").open("w") as f:
        for entry in index: f.write(__import__("json").dumps(entry) + "\n")
    keys = ["frame_id", "station_id", "pilot", "unknown_capture_only", "valid_fraction",
            "alpha_mean", "render_seconds", "flags"]
    with (out / "quality.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(qa_rows)
    summary = {"plan_hash": plan["plan_hash"], "planned": len(plan["frames"]), "complete": len(complete),
               "pending": pending, "review_flagged": sum(bool(q["flags"]) for q in qa_rows),
               "note": "Do not auto-discard flagged views. Inspect RGB/depth; flags are only heuristics."}
    write_json(out / "summary.json", summary)
    qa_dir = out / "qa"; qa_dir.mkdir(exist_ok=True)
    pages = []
    for chunk in range(0, len(complete), 36):
        group = complete[chunk:chunk+36]
        columns = 6; tw, th = 180, 135; lh = 36
        rows = math.ceil(len(group) / columns)
        for mode in ("rgb", "depth"):
            sheet = Image.new("RGB", (columns*tw, rows*(th+lh)), (32, 37, 45))
            draw = ImageDraw.Draw(sheet)
            for j, v in enumerate(group):
                folder = out / "frames" / v["frame_id"]
                if mode == "rgb":
                    with Image.open(folder / "rgb.png") as im: thumb = im.convert("RGB").resize((tw, th))
                else:
                    d = np.load(folder / "depth_m.npy", allow_pickle=False)
                    with Image.open(folder / "valid.png") as im: valid = np.array(im) > 0
                    normalized = np.where(valid, 1 - np.clip(d/8., 0, 1), 0)
                    arr = np.rint(np.nan_to_num(normalized)*255).astype(np.uint8)
                    thumb = Image.fromarray(arr).convert("RGB").resize((tw, th))
                x, y = (j % columns)*tw, (j // columns)*(th+lh)
                sheet.paste(thumb, (x,y))
                draw.text((x+3, y+th+2), v["frame_id"], fill="white")
                draw.text((x+3, y+th+15), f"yaw {v['yaw_deg']:.0f} pitch {v['pitch_deg']:.0f}" +
                          (" GRAY" if v["unknown_capture_only"] else ""), fill="white")
            name = f"contact_{mode}_{chunk//36+1:03d}.jpg"
            sheet.save(qa_dir / name, quality=90)
            pages.append(name)
    links = "\n".join(f'<p>{n}</p><a href="{n}"><img src="{n}" style="max-width:100%"></a>' for n in pages)
    (qa_dir / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Capture review</title>"
                                     "<h1>Valmet calibrated-view review</h1>"
                                     f"<p>{len(complete)}/{len(plan['frames'])} frames complete. "
                                     "Depth thumbnails: 0-8 provisional m, near bright; invalid black.</p>" + links)
    print(f"Report: {len(complete)}/{len(plan['frames'])} complete, {summary['review_flagged']} review flags. "
          f"Open {qa_dir/'index.html'}", flush=True)
