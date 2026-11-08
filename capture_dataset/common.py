"""CPU-only planning, metadata and persistence; imports the existing avatar conventions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from avatar_demo.core import FloorMap, camera_matrices

SCHEMA = "valmet.calibrated_views.v1"
ROTATION = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
REQUIRED_FRAME_FILES = ("rgb.png", "depth_scene.npy", "depth_m.npy", "alpha.npy",
                        "valid.png", "camera.json", "qa.json")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def map_identity(path):
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text())
    image = Path(str(cfg["image"])).expanduser()
    if not image.is_absolute():
        image = path.parent / image
    return {"yaml": str(path), "yaml_sha256": file_sha256(path),
            "image": str(image.resolve()), "image_sha256": file_sha256(image)}


def candidate_clearance(floor_map):
    free = floor_map.occupancy == 0
    return distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * floor_map.resolution


def select_stations(floor_map, pins, excluded, count=40, clearance=0.10,
                    min_spacing=0.50):
    """Pinned unknown cells are capture-only exceptions, never navigation edits.

    Fill remaining stations by deterministic Euclidean farthest-point sampling
    over candidate known-free cells. This does not establish visibility or reachability.
    """
    if count < 1 or count > 500 or clearance < 0 or min_spacing <= 0:
        raise ValueError("Invalid station count, clearance or spacing.")
    if len(pins) > count:
        raise ValueError("More pins than requested stations: increase --positions or remove pins.")
    stations = []
    used = set()
    for p in pins:
        xy = np.asarray([p["x"], p["y"]], float)
        if not np.isfinite(xy).all():
            raise ValueError("Pinned coordinates must be finite.")
        cell = floor_map.world_to_cell(xy)
        if not floor_map.contains(cell):
            raise ValueError("Pinned camera is outside the map.")
        state = int(floor_map.occupancy[cell])
        if state not in (0, -1):
            raise ValueError("Occupied cells cannot be pinned in this capture tool.")
        if state == -1 and not p.get("allow_unknown", False):
            raise ValueError("Unknown pin requires explicit capture-only approval.")
        if cell in used:
            raise ValueError("Duplicate pin in the same map cell.")
        used.add(cell)
        # Snap to cell center so the editor and dataset use identical coordinates.
        xy = floor_map.cell_to_world(cell)
        stations.append({"x": float(xy[0]), "y": float(xy[1]),
                         "row": int(cell[0]), "col": int(cell[1]),
                         "selection": "manual_pin", "map_state": state,
                         "unknown_capture_only": state == -1})
    distances = candidate_clearance(floor_map)
    allowed = (floor_map.occupancy == 0) & (distances >= clearance)
    rr, cc = np.indices(floor_map.shape)
    all_xy = floor_map.cells_to_world(np.column_stack([rr.ravel(), cc.ravel()]))
    for e in excluded:
        ex, ey, radius = float(e["x"]), float(e["y"]), float(e["radius"])
        if not np.isfinite([ex, ey, radius]).all() or radius <= 0:
            raise ValueError("Invalid exclusion circle.")
        allowed &= (np.linalg.norm(all_xy - [ex, ey], axis=1).reshape(allowed.shape) > radius)
        if any(math.hypot(p["x"] - ex, p["y"] - ey) <= radius for p in stations):
            raise ValueError("An exclusion overlaps a pin. Remove that pin or exclusion.")
    cells = np.argwhere(allowed)
    coords = floor_map.cells_to_world(cells) if len(cells) else np.empty((0, 2))
    if stations:
        nearest = np.min([np.linalg.norm(coords - [p["x"], p["y"]], axis=1)
                          for p in stations], axis=0)
    else:
        nearest = np.full(len(cells), np.inf)
    while len(stations) < count and len(cells):
        if not stations:
            idx = int(np.argmax(distances[cells[:, 0], cells[:, 1]]))
        else:
            idx = int(np.argmax(nearest))
            if nearest[idx] < min_spacing:
                break
        cell, xy = cells[idx], coords[idx]
        stations.append({"x": float(xy[0]), "y": float(xy[1]),
                         "row": int(cell[0]), "col": int(cell[1]),
                         "selection": "auto_known_free", "map_state": 0,
                         "unknown_capture_only": False})
        nearest = np.minimum(nearest, np.linalg.norm(coords - xy, axis=1))
        nearest[idx] = -np.inf
    if not stations:
        raise ValueError("No capture positions. Inspect the map or lower --clearance.")
    for i, p in enumerate(stations):
        p["station_id"] = f"s{i:03d}"
        p["navigation_validated"] = False
    return stations


def spread_subset(stations, count):
    if not stations or count <= 0:
        return []
    coords = np.array([[p["x"], p["y"]] for p in stations])
    chosen = [0]
    dist = np.linalg.norm(coords - coords[0], axis=1)
    dist[0] = -1
    while len(chosen) < min(count, len(stations)):
        idx = int(np.argmax(dist))
        chosen.append(idx)
        dist = np.minimum(dist, np.linalg.norm(coords - coords[idx], axis=1))
        dist[chosen] = -1
    return [stations[i]["station_id"] for i in chosen]


def pilot_station_ids(stations):
    pins = [s for s in stations if s["selection"] == "manual_pin"]
    autos = [s for s in stations if s["selection"] == "auto_known_free"]
    n_pins = min(3, len(pins))
    ids = spread_subset(pins, n_pins) + spread_subset(autos, 6 - n_pins)
    if len(ids) < min(6, len(stations)):
        ids += spread_subset([s for s in stations if s["station_id"] not in ids], 6 - len(ids))
    return ids


def create_plan(map_path, scene_path, stations, pins, excluded, cfg):
    fmap = FloorMap.load(map_path)
    pilots = pilot_station_ids(stations)
    views = []
    for s in stations:
        for yi, yaw in enumerate(cfg["yaws_deg"]):
            for pi, pitch in enumerate(cfg["pitches_deg"]):
                frame_id = f"{s['station_id']}_y{yi:02d}_p{pi:02d}"
                views.append({"frame_id": frame_id, "station_id": s["station_id"],
                              "x": s["x"], "y": s["y"], "yaw_deg": float(yaw),
                              "pitch_deg": float(pitch), "camera_height_m": cfg["camera_height_m"],
                              "map_state": s["map_state"], "selection": s["selection"],
                              "unknown_capture_only": s["unknown_capture_only"],
                              "pilot": s["station_id"] in pilots and yi % 2 == 0})
    plan = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": {"map": map_identity(map_path),
                                         "scene_path": str(Path(scene_path).expanduser().resolve())},
            "config": cfg, "map": {"shape": list(fmap.shape), "origin": fmap.origin.tolist(),
                                    "resolution_m": fmap.resolution},
            "pins": pins, "excluded_capture_disks": excluded, "stations": stations,
            "frames": views, "pilot_station_ids": pilots,
            "notes": ["Synthetic stationary viewpoints, not a navigated trajectory.",
                      "Manual unknown pins authorize rendering only; original map is unchanged.",
                      "Scale and flat floor are provisional. No 3D camera/body collision check.",
                      "Capture positions are spread spatially; scene/object visibility is not guaranteed."]}
    plan["plan_hash"] = digest(plan)
    return plan


def load_plan(out):
    plan = read_json(Path(out) / "plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    if plan.get("schema") != SCHEMA or digest(body) != plan.get("plan_hash"):
        raise ValueError("Plan hash/schema mismatch. Create a new dataset instead of editing a committed plan.")
    return plan


def camera_record(plan, view):
    cfg = plan["config"]
    K, c2g, g2c = camera_matrices(view["x"], view["y"], math.radians(view["yaw_deg"]),
                                 view["camera_height_m"], view["pitch_deg"],
                                 scale=cfg["scale"], floor_y=cfg["floor_y_scene"],
                                 width=cfg["width"], height_px=cfg["height"], fov_y_deg=cfg["fov_y_deg"])
    # Camera-to-map is rigid because its translation and depth are expressed in metres.
    # Do NOT multiply orientation columns by scale.
    s = cfg["scale"]
    translation = np.array([0., 0., s * cfg["floor_y_scene"]])
    c2m = np.eye(4)
    c2m[:3, :3] = ROTATION @ c2g[:3, :3]
    c2m[:3, 3] = s * ROTATION @ c2g[:3, 3] + translation
    return {"schema": SCHEMA, "frame_id": view["frame_id"], "station_id": view["station_id"],
            "plan_hash": plan["plan_hash"], "width": cfg["width"], "height": cfg["height"],
            "K": K.tolist(), "camera_to_world": c2g.tolist(), "world_to_camera": g2c.tolist(),
            "camera_to_map": c2m.tolist(), "map_to_camera": np.linalg.inv(c2m).tolist(),
            "world_frame": "original_Gaussian_scene", "map_frame": "existing_ROS_map_provisional_metres",
            "coordinate_convention": "OpenCV: +X right, +Y down, +Z forward",
            "depth_type": "expected_camera_z", "depth_scene_file": "depth_scene.npy",
            "depth_metres_file": "depth_m.npy", "depth_metres_is_derived": True,
            "depth_scale_scene_to_metres": s,
            "pixel_sampling": {"u": "column + 0.5", "v": "row + 0.5"},
            "map_pose": view,
            "scene_to_map": {"scale": s, "rotation": ROTATION.tolist(),
                             "translation_m": translation.tolist(), "floor_y_scene": cfg["floor_y_scene"],
                             "equation": "p_map = scale * R * p_scene + translation_m"},
            "raw_depth_retained": True,
            "validity_mask_rule": f"finite and positive raw depth, finite alpha >= {cfg['alpha_min']}",
            "geometry_note": "Expected splat depth and alpha coverage are not verified physical geometry/confidence."}


def map_image(fmap, max_size=(1000, 800)):
    colors = np.empty((*fmap.shape, 3), dtype=np.uint8)
    colors[:] = (104, 113, 125)
    colors[fmap.occupancy == 100] = (35, 42, 52)
    colors[fmap.occupancy == 0] = (201, 224, 188)
    image = Image.fromarray(np.flipud(colors))
    ratio = min(max_size[0] / image.width, max_size[1] / image.height)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    return image.resize(size, Image.Resampling.NEAREST)


def world_to_pixel(fmap, xy, size):
    col, row = fmap.world_to_grid_continuous(xy)
    return col / fmap.shape[1] * size[0], (1 - row / fmap.shape[0]) * size[1]


def pixel_to_world(fmap, xy, size):
    col = xy[0] / size[0] * fmap.shape[1]
    row = (1 - xy[1] / size[1]) * fmap.shape[0]
    return fmap.origin[:2] + fmap.rotation @ (np.array([col, row]) * fmap.resolution)


def positions_image(fmap, stations, excluded=(), pilots=(), max_size=(1000, 800)):
    base = map_image(fmap, max_size=max_size)
    canvas = Image.new("RGB", (base.width, base.height + 62), "white")
    canvas.paste(base, (0, 62))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), "Capture positions only | purple=pinned unknown, gold=pinned free, blue=automatic", fill="black")
    draw.text((8, 27), "P = pilot station. All locations require rendered-view inspection. Navigation map is NOT changed.", fill="black")
    scale_px = base.width / (fmap.shape[1] * fmap.resolution)
    for ex in excluded:
        x, y = world_to_pixel(fmap, [ex["x"], ex["y"]], base.size); y += 62
        r = ex["radius"] * scale_px
        draw.ellipse((x-r, y-r, x+r, y+r), outline="#ff7777", width=2)
    for p in stations:
        x, y = world_to_pixel(fmap, [p["x"], p["y"]], base.size); y += 62
        color = "#bd65e8" if p["map_state"] == -1 else ("#e7ae32" if p["selection"] == "manual_pin" else "#2476da")
        draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline="white", width=1)
        label = p["station_id"] + (" P" if p["station_id"] in pilots else "")
        draw.text((x+7, y-6), label, fill="white", stroke_width=1, stroke_fill="#202020")
    return canvas


def save_plan(out, plan):
    out = Path(out)
    if (out / "plan.json").exists() or (out / "frames").exists():
        raise FileExistsError(f"Dataset already planned: {out}. Use a new directory for revised poses/settings.")
    out.mkdir(parents=True, exist_ok=True)
    fmap = FloorMap.load(plan["inputs"]["map"]["yaml"])
    positions_image(fmap, plan["stations"], plan["excluded_capture_disks"],
                    plan["pilot_station_ids"]).save(out / "camera_positions.png")
    with (out / "camera_positions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(plan["stations"][0]))
        writer.writeheader(); writer.writerows(plan["stations"])
    with (out / "frames.jsonl").open("w") as f:
        for frame in plan["frames"]:
            f.write(json.dumps(frame) + "\n")
    write_json(out / "plan.json", plan)


def quality_report(rgb, depth_m, alpha, alpha_min):
    valid = np.isfinite(depth_m) & (depth_m > 0) & np.isfinite(alpha) & (alpha >= alpha_min)
    values = depth_m[valid]
    near = float(np.mean(values < 0.20)) if len(values) else None
    percentiles = np.percentile(values, [1, 10, 50, 90, 99]).tolist() if len(values) else None
    flags = []
    if valid.mean() < 0.80: flags.append("low_render_coverage")
    if near is not None and near > 0.25: flags.append("many_surfaces_within_0.2m_check_camera")
    if float(np.std(rgb.astype(np.float32))) < 8: flags.append("low_rgb_variation")
    return {"valid_fraction": float(valid.mean()), "alpha_mean": (float(alpha[np.isfinite(alpha)].mean()) if np.isfinite(alpha).any() else None),
            "depth_m_percentiles_1_10_50_90_99": percentiles, "near_fraction_below_0_2m": near,
            "flags": flags, "auto_rejected": False,
            "note": "Heuristic review flags only. High alpha is not evidence of geometric correctness."}, valid


def frame_complete(folder, plan_hash):
    folder = Path(folder)
    try:
        stamp = read_json(folder / "complete.json")
        if stamp["plan_hash"] != plan_hash: return False
        for name in REQUIRED_FRAME_FILES:
            p = folder / name
            if not p.is_file() or p.stat().st_size != stamp["files"][name]["bytes"]:
                return False
            if file_sha256(p) != stamp["files"][name]["sha256"]: return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def save_frame(folder, plan, view, rgb, depth_scene, alpha, seconds=0.):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "complete.json").unlink(missing_ok=True)
    cfg = plan["config"]
    shape = (cfg["height"], cfg["width"])
    if rgb.shape != (*shape, 3) or depth_scene.shape != shape or alpha.shape != shape:
        raise ValueError("Rendered dimensions do not match plan.")
    depth_scene = np.asarray(depth_scene, dtype=np.float32)
    depth_m = depth_scene * np.float32(cfg["scale"])
    alpha = np.asarray(alpha, dtype=np.float32)
    qa, valid = quality_report(rgb, depth_m, alpha, cfg["alpha_min"])
    qa["render_seconds"] = float(seconds)
    Image.fromarray(rgb).save(folder / "rgb.png")
    np.save(folder / "depth_scene.npy", depth_scene)
    np.save(folder / "depth_m.npy", depth_m)
    np.save(folder / "alpha.npy", alpha)
    Image.fromarray(valid.astype(np.uint8) * 255).save(folder / "valid.png")
    camera = camera_record(plan, view)
    camera["saved_utc"] = datetime.now(timezone.utc).isoformat()
    camera["render_seconds"] = float(seconds)
    write_json(folder / "camera.json", camera)
    write_json(folder / "qa.json", qa)
    files = {n: {"bytes": (folder/n).stat().st_size, "sha256": file_sha256(folder/n)}
             for n in REQUIRED_FRAME_FILES}
    write_json(folder / "complete.json", {"frame_id": view["frame_id"],
                                            "plan_hash": plan["plan_hash"], "files": files})
