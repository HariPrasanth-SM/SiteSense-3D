"""Persistent CUDA renderer in a spawned process, isolated from Tk's event loop."""
from __future__ import annotations

import json
import math
import queue
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .core import camera_matrices


def save_frame(frame: dict, prefix: str | Path):
    """Preserve the ACTUAL rendered pose, RGB, alpha, and both depth unit conventions."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame["rgb"]).save(str(prefix) + "_rgb.png")
    np.save(str(prefix) + "_depth.npy", frame["depth_scene"].astype(np.float32))
    np.save(str(prefix) + "_depth_m.npy", frame["depth_m"].astype(np.float32))
    np.save(str(prefix) + "_alpha.npy", frame["alpha"].astype(np.float32))
    with open(str(prefix) + "_camera.json", "w") as stream:
        json.dump(frame["camera"], stream, indent=2)


def _load_scene(filename, torch, sh_degree):
    # Import only in the worker: map-only operation doesn't need plyfile/torch/gsplat.
    from plyfile import PlyData
    ply = PlyData.read(filename, mmap=True)
    v = ply["vertex"].data
    names = set(v.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    required |= {f"rot_{i}" for i in range(4)}
    required |= {f"scale_{i}" for i in range(3)}
    required |= {f"f_dc_{i}" for i in range(3)}
    if not required <= names:
        raise ValueError("Expected the ORIGINAL Gaussian-splat PLY, not the exported ROS point cloud. "
                         f"Missing fields: {sorted(required - names)}")
    def columns(keys):
        return np.column_stack([np.asarray(v[k], dtype=np.float32) for k in keys])
    means = columns(["x", "y", "z"])
    logs = columns([f"scale_{i}" for i in range(3)])
    quats = columns([f"rot_{i}" for i in range(4)])  # Explicit w,x,y,z field names.
    logits = np.asarray(v["opacity"], dtype=np.float32)
    dc = columns([f"f_dc_{i}" for i in range(3)])[:, None, :]
    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if len(rest_names) % 3:
        raise ValueError("Unexpected SH coefficient count.")
    available_bases = 1 + len(rest_names) // 3
    if (sh_degree + 1) ** 2 > available_bases:
        raise ValueError(f"Requested SH degree {sh_degree}, file only has {available_bases} bases.")
    if sh_degree:
        rest = columns(rest_names).reshape(len(means), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest[:, : (sh_degree + 1) ** 2 - 1]], axis=1)
        del rest
    else:
        sh = dc
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    valid = (np.isfinite(means).all(1) & np.isfinite(logs).all(1) & np.isfinite(quats).all(1)
             & np.isfinite(logits) & np.isfinite(sh).all(axis=(1, 2)) & (norms[:, 0] > 1e-8))
    if not valid.any():
        raise ValueError("No finite Gaussian parameters.")
    if not valid.all():
        means, logs, quats, logits, sh, norms = [a[valid] for a in (means, logs, quats, logits, sh, norms)]
    quats = quats / norms
    scales = np.exp(np.clip(logs, -30, 20)).astype(np.float32)
    opacities = (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32)
    def cuda(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to("cuda")
    result = tuple(cuda(a) for a in (means, quats, scales, opacities, sh))
    return result


def _send_status(q, kind, text):
    try:
        q.put_nowait((kind, text))
    except queue.Full:
        pass


def render_loop(config, shared_pose, frames, statuses, stop_event):
    """shared_pose = [sequence, x_map, y_map, yaw, height_m, pitch_deg, simulation_time]."""
    try:
        _send_status(statuses, "loading", "Loading the original Gaussian scene once on the GPU...")
        import torch
        from gsplat import rasterization
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment. Use your existing gsplat .venv.")
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        scene = _load_scene(config["scene"], torch, config["sh_degree"])
        _send_status(statuses, "ready", f"Loaded {len(scene[0]):,} Gaussians. Waiting for camera pose.")
        last_sequence = -1
        next_allowed = 0.0
        frame_number = 0
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_allowed:
                stop_event.wait(min(0.05, next_allowed - now))
                continue
            with shared_pose.get_lock():
                values = list(shared_pose[:])
            seq, x, y, yaw, camera_height, pitch, sim_t = values
            if seq <= last_sequence:
                stop_event.wait(0.02)
                continue
            last_sequence = seq
            started = time.monotonic()
            K, c2w, w2c = camera_matrices(
                x, y, yaw, camera_height, pitch,
                scale=config["scale"], floor_y=config["floor_y"],
                width=config["width"], height_px=config["height"], fov_y_deg=config["fov"],
            )
            with torch.inference_mode():
                rendered, alphas, meta = rasterization(
                    means=scene[0], quats=scene[1], scales=scene[2], opacities=scene[3], colors=scene[4],
                    viewmats=torch.from_numpy(w2c[None]).to("cuda"),
                    Ks=torch.from_numpy(K[None]).to("cuda"),
                    width=config["width"], height=config["height"],
                    sh_degree=config["sh_degree"], render_mode="RGB+ED", packed=True,
                    near_plane=0.03 / config["scale"], far_plane=60.0 / config["scale"],
                )
                pixels = rendered[0].detach().cpu().numpy()
                alpha = alphas[0, :, :, 0].detach().cpu().numpy().copy()
                rgb = np.rint(np.clip(pixels[:, :, :3], 0, 1) * 255).astype(np.uint8)
                depth_scene = pixels[:, :, 3].copy().astype(np.float32)
                del rendered, alphas, meta, pixels
            elapsed = time.monotonic() - started
            # Alpha is rendering coverage, not a geometric confidence estimate.
            valid = np.isfinite(depth_scene) & (depth_scene > 0) & (alpha >= config["alpha_min"])
            depth_scene[~valid] = np.nan
            depth_m = depth_scene * config["scale"]
            camera = {
                "K": K.tolist(), "camera_to_world": c2w.tolist(), "world_to_camera": w2c.tolist(),
                "width": config["width"], "height": config["height"],
                "coordinate_convention": "OpenCV", "world_frame": "original_Gaussian_scene",
                "depth_type": "expected_camera_z", "depth_units": "scene units in *_depth.npy; provisional metres in *_depth_m.npy",
                "scene_to_map_scale": config["scale"], "floor_y_scene": config["floor_y"],
                "map_pose": {"x": x, "y": y, "yaw_rad": yaw, "camera_height_m": camera_height, "pitch_deg": pitch},
                "sequence": int(seq), "simulation_time": sim_t,
                "render_seconds": elapsed, "captured_wall_time": time.time(),
                "geometry_note": "Expected Gaussian depth, not verified surface geometry or a physical sensor measurement.",
            }
            frame = {
                "rgb": rgb, "depth_scene": depth_scene, "depth_m": depth_m, "alpha": alpha,
                "camera": camera, "pose": [x, y, yaw], "sequence": int(seq),
                "render_seconds": elapsed, "requested_at": started, "finished_at": time.monotonic(),
            }
            if config["record"]:
                save_frame(frame, Path(config["session"]) / "frames" / f"frame_{frame_number:06d}")
                frame_number += 1
            try:
                frames.put(frame, timeout=0.03)
            except queue.Full:
                # Retry this pose if the GUI was busy. This also ensures that
                # the final stationary/facing-target frame is not lost.
                last_sequence = seq - 1
            next_allowed = started + 1.0 / config["render_hz"]
    except BaseException as exc:
        text = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print("\nRENDER WORKER ERROR:\n" + text, flush=True)
        _send_status(statuses, "error", text)
    finally:
        _send_status(statuses, "stopped", "Renderer stopped; map-only controls remain available.")
