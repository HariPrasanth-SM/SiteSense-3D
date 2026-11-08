"""Seeded multi-height view planning and explicit capture-only corridor sampling.

No source map is edited; no 3-D clearance or object visibility is inferred.
The saved frame schema is compatible with the existing capture_dataset renderer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from PIL import ImageDraw, ImageFont

from .clearance import required_clearance, clearance_layers, pin_clearance

from capture_dataset.common import (
    SCHEMA, FloorMap, digest, file_sha256,
    map_identity, map_image, pixel_to_world, read_json, save_plan as base_save_plan,
    select_stations as base_select_stations, spread_subset, world_to_pixel, write_json,
)

# Bounds refer to central optical-axis pitch, not the entire field of view.
PITCH_POLICY = [
    {"height_lt_m": 0.9, "pitch_min_deg": -5., "pitch_max_deg": 0.},
    {"height_lt_m": 1.5, "pitch_min_deg": -12., "pitch_max_deg": 4.},
    {"height_lt_m": 2.0, "pitch_min_deg": -20., "pitch_max_deg": 2.},
    {"height_lt_m": None, "pitch_min_deg": -40., "pitch_max_deg": -10.},
]
HEADER = 98


def load_selection(path, map_path):
    if path is None:
        return {"pins": [], "corridors": [], "excluded_capture_disks": []}
    saved = read_json(path)
    if saved.get("map_identity") != map_identity(map_path):
        raise ValueError("Selection belongs to a different map. Use the same map YAML/image.")
    return {"pins": saved.get("pins", []), "corridors": saved.get("corridors", []),
            "excluded_capture_disks": saved.get("excluded_capture_disks", [])}


def segment_cells(fmap, a, b):
    """Supercover a 2-D line, including corner contacts (capture approval only)."""
    p, q = fmap.world_to_grid_continuous(a), fmap.world_to_grid_continuous(b)
    if not np.isfinite([*p, *q]).all():
        raise ValueError("Corridor coordinates must be finite.")
    delta, times = q - p, [0., 1.]
    for axis in range(2):
        if abs(delta[axis]) > 1e-12:
            lo, hi = sorted([float(p[axis]), float(q[axis])])
            for edge in range(math.floor(lo) + 1, math.ceil(hi)):
                t = (edge - p[axis]) / delta[axis]
                if 0 < t < 1:
                    times.append(float(t))
    times = sorted(set(times))
    probes = sorted(times + [(a + b) / 2 for a, b in zip(times[:-1], times[1:])])
    seen = set()
    for t in probes:
        col, row = p + t * delta
        def indices(value):
            k = round(float(value))
            return (k - 1, k) if abs(value - k) < 1e-9 else (math.floor(value),)
        for r in indices(row):
            for c in indices(col):
                if (r, c) not in seen:
                    seen.add((r, c))
                    yield (r, c)


def validate_segment(fmap, a, b, allow_unknown):
    for cell in segment_cells(fmap, a, b):
        if not fmap.contains(cell):
            raise ValueError("Corridor leaves the map. Add intermediate points within the passage.")
        state = int(fmap.occupancy[cell])
        if state == 100:
            raise ValueError("Corridor line crosses an OCCUPIED cell. Undo and add bends along the passage.")
        if state == -1 and not allow_unknown:
            raise ValueError("Corridor crosses GRAY cells. Enable capture-only gray permission first.")


def corridor_pins(fmap, corridors, spacing):
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Corridor spacing must be positive.")
    result, used = [], set()
    for i, corridor in enumerate(corridors):
        pts = np.asarray(corridor.get("points", []), dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all():
            raise ValueError("Each corridor needs at least two finite [x,y] vertices.")
        allow = bool(corridor.get("allow_unknown", False))
        cid = f"c{i:03d}"
        for a, b in zip(pts[:-1], pts[1:]):
            validate_segment(fmap, a, b, allow)
            n = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
            for t in np.linspace(0., 1., n + 1):
                cell = fmap.world_to_cell(a + t * (b - a))
                if cell in used:
                    continue
                used.add(cell)
                xy = fmap.cell_to_world(cell)
                state = int(fmap.occupancy[cell])
                result.append({"x": float(xy[0]), "y": float(xy[1]),
                               "allow_unknown": state == -1,
                               "corridor_id": cid,
                               "note": "Interpolated user-marked capture-only corridor, not a navigation path."})
    return result


def choose_stations(fmap, selection, cfg, diagnostics=None):
    """Apply a map-edge buffer to every station, including pins/corridors.

    Approved pins/lines are never silently relocated. Close samples are skipped
    and recorded; remaining budget is filled with buffered automatic free cells.
    Unknown cells still require explicit user permission.
    """
    required = required_clearance(cfg)
    layers = clearance_layers(fmap)
    pins = selection.get("pins", [])
    cpins = corridor_pins(fmap, selection.get("corridors", []), cfg["corridor_spacing_m"])
    merged, source, used, rejected = [], {}, set(), []
    corridor_members = {fmap.world_to_cell([p["x"], p["y"]]): p["corridor_id"] for p in cpins}
    for origin, items in (("manual_pin", pins), ("corridor_sample", cpins)):
        for pin in items:
            cell, detail = pin_clearance(fmap, pin, layers, required)
            if cell in used:
                continue
            used.add(cell)
            if not detail["accepted"]:
                rejected.append({**detail, "selection": origin,
                                 "corridor_id": corridor_members.get(cell),
                                 "reason": "closer than requested clearance to occupied cell or map exterior"})
                continue
            merged.append(dict(pin))
            source[cell] = origin
    if len(merged) > cfg["requested_positions"]:
        raise ValueError(f"Pins/corridors require {len(merged)} positions, exceeding budget "
                         f"{cfg['requested_positions']}. Increase --positions or --corridor-spacing.")
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"required_clearance_m": required,
                            "distance_is_conservative_map_edge_lower_bound": True,
                            "cell_edge_allowance_m": layers["cell_edge_allowance_m"],
                            "rejected_explicit_samples": rejected,
                            "rejected_count": len(rejected),
                            "manual_pins_kept": sum(v == "manual_pin" for v in source.values()),
                            "corridor_samples_kept": sum(v == "corridor_sample" for v in source.values()),
                            "policy": "All stations buffered from occupied cells and map exterior. "
                                      "Automatic stations additionally buffered from unknown. Explicit "
                                      "pins/corridors may approach unknown; gray positions require opt-in. "
                                      "No relocation, 3-D clearance, or navigation approval."})
    try:
        stations = base_select_stations(
            fmap, merged, selection.get("excluded_capture_disks", []), cfg["requested_positions"],
            required + layers["cell_edge_allowance_m"], cfg["auto_spacing_m"])
    except ValueError as exc:
        if str(exc).startswith("No capture positions"):
            raise ValueError(f"No camera positions satisfy the {required:.2f} m map buffer. "
                             "Review the map or use an explicit smaller clearance in a NEW plan; "
                             "the tool will not silently reduce it.") from exc
        raise
    for station in stations:
        cell = (station["row"], station["col"])
        station["selection"] = source.get(cell, "auto_known_free")
        station["corridor_id"] = corridor_members.get(cell)
        station["wall_clearance_lower_bound_m"] = float(layers["wall_m"][cell])
        station["known_free_clearance_lower_bound_m"] = float(layers["known_m"][cell])
        station["required_clearance_m"] = required
        if station["wall_clearance_lower_bound_m"] + 1e-12 < required:
            raise AssertionError("Internal error: generated camera violates spawn clearance.")
    if diagnostics is not None:
        diagnostics["minimum_retained_wall_clearance_m"] = min(
            s["wall_clearance_lower_bound_m"] for s in stations)
        diagnostics["actual_positions"] = len(stations)
    return stations


def pitch_bounds(height):
    for band in PITCH_POLICY:
        if band["height_lt_m"] is None or height < band["height_lt_m"]:
            return band["pitch_min_deg"], band["pitch_max_deg"]
    raise AssertionError("Unreachable pitch policy")


def choose_pilots(stations, count=8):
    """Include corridor/gray samples, explicit pins and spatially spread room samples."""
    count = min(count, len(stations))
    priority = [s for s in stations if s["unknown_capture_only"] or s.get("corridor_id")]
    selected = spread_subset(priority, min(4, count, len(priority)))
    pins = [s for s in stations if s["selection"] == "manual_pin" and s["station_id"] not in selected]
    selected += spread_subset(pins, min(2, count - len(selected), len(pins)))
    others = [s for s in stations if s["station_id"] not in selected]
    selected += spread_subset(others, count - len(selected))
    return selected


def randomized_views(stations, cfg):
    """Stratified heights and headings; fixed seed, no uncontrolled randomness at render time."""
    total = cfg["requested_frames"]
    if total < len(stations):
        raise ValueError("Frame budget must be at least the actual number of stations.")
    counts = np.full(len(stations), total // len(stations), dtype=int)
    # Distribute remainder without always favoring the first/pinned positions.
    global_rng = np.random.default_rng(cfg["seed"])
    extra = global_rng.permutation(len(stations))[:total % len(stations)]
    counts[extra] += 1
    pilot_ids = choose_pilots(stations, cfg["pilot_positions"])
    views = []
    lo, hi = cfg["height_range_m"]
    for si, station in enumerate(stations):
        n = int(counts[si])
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], si, 731]))
        # Every station samples the full height range and a full turn of headings.
        height_order = rng.permutation(n)
        heights = lo + ((height_order + rng.uniform(.05, .95, n)) / n) * (hi - lo)
        yaw_bins = rng.permutation(n)
        yaw_phase = float(rng.uniform(0, 360))
        yaws = (yaw_phase + (yaw_bins + rng.uniform(.10, .90, n)) * (360. / n)) % 360
        order = np.argsort(heights)
        # Six views spanning low/mid/high heights, not simply the first six frames.
        pilot_indices = set(int(order[j]) for j in np.unique(
            np.rint(np.linspace(0, n - 1, min(cfg["pilot_views_per_position"], n))).astype(int)))
        for j in range(n):
            h = float(heights[j])
            pmin, pmax = pitch_bounds(h)
            view = {"frame_id": f"{station['station_id']}_v{j:03d}",
                    "station_id": station["station_id"], "view_index": j,
                    "x": station["x"], "y": station["y"],
                    "yaw_deg": float(yaws[j]), "pitch_deg": float(rng.uniform(pmin, pmax)),
                    "camera_height_m": h, "height_stratum": int(height_order[j]),
                    "yaw_sector": int(yaw_bins[j]), "yaw_phase_deg": yaw_phase,
                    "pitch_limits_deg": [pmin, pmax],
                    "map_state": station["map_state"], "selection": station["selection"],
                    "corridor_id": station.get("corridor_id"),
                    "spawn_wall_clearance_lower_bound_m": station.get("wall_clearance_lower_bound_m"),
                    "spawn_required_clearance_m": station.get("required_clearance_m"),
                    "unknown_capture_only": station["unknown_capture_only"],
                    "pilot": station["station_id"] in pilot_ids and j in pilot_indices}
            views.append(view)
    return views, pilot_ids


def create_dense_plan(map_path, scene_path, stations, selection, cfg):
    fmap = FloorMap.load(map_path)
    audit = {}
    checked_stations = choose_stations(fmap, selection, cfg, diagnostics=audit)
    if checked_stations != stations:
        raise ValueError("Stations differ from the current clearance plan. Recompute before saving.")
    views, pilots = randomized_views(stations, cfg)
    plan = {"schema": SCHEMA, "generator": "capture_dataset_dense/2.1.0",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {"map": map_identity(map_path),
                       "scene_path": str(Path(scene_path).expanduser().resolve())},
            "config": cfg,
            "map": {"shape": list(fmap.shape), "origin": fmap.origin.tolist(), "resolution_m": fmap.resolution},
            "pins": selection.get("pins", []),
            "corridors": selection.get("corridors", []),
            "excluded_capture_disks": selection.get("excluded_capture_disks", []),
            "stations": stations, "frames": views, "pilot_station_ids": pilots,
            "spawn_clearance": audit,
            "notes": ["Independent synthetic views, not avatar movement or physical captures.",
                      "Corridor drawing grants capture permission only; no source occupancy edits.",
                      "Heights/scale assume floor Gaussian Y=1.1 and provisional scale 1.43 unless configured otherwise.",
                      "Height-dependent pitch avoids deliberately pointing toward the ceiling; ceiling pixels may remain in the FOV.",
                      "Map-only clearance enforced on all stations; rejected explicit samples are logged, not moved.",
                      "No 3-D camera clearance or object-coverage guarantee. Review pilot and quality flags."],
            "planner_source_sha256": file_sha256(__file__),
            "clearance_source_sha256": file_sha256(Path(__file__).with_name("clearance.py"))}
    plan["plan_hash"] = digest(plan)
    return plan


def preview(fmap, stations, selection, pilots=(), max_size=(1100, 720), all_labels=False, cfg=None, diagnostics=None):
    """Readable dense overview: label pilot/corridor stations, with optional full IDs."""
    from PIL import Image
    base = map_image(fmap, max_size)
    if cfg is not None:
        # A visual overlay only; source occupancy and selection are not modified.
        bounds = clearance_layers(fmap)
        required = required_clearance(cfg)
        hard_buffer = (fmap.occupancy != 100) & (bounds["wall_m"] + 1e-12 < required)
        auto_buffer = (fmap.occupancy == 0) & (bounds["known_m"] + 1e-12 < required) & ~hard_buffer
        overlay = np.zeros((*fmap.shape, 4), dtype=np.uint8)
        overlay[auto_buffer] = [180, 145, 65, 150]
        overlay[hard_buffer] = [205, 80, 65, 170]
        mask_image = Image.fromarray(np.flipud(overlay)).resize(base.size, Image.Resampling.NEAREST)
        base = Image.alpha_composite(base.convert("RGBA"), mask_image).convert("RGB")
    image = Image.new("RGB", (base.width, base.height + HEADER), "white")
    image.paste(base, (0, HEADER))
    d = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    d.text((8, 6), "DENSE CAPTURE | blue: auto | gold: pin | cyan: corridor | purple: approved gray", fill="black", font=font)
    d.text((8, 26), f"{len(stations)} stations. P = pilot. Gray permission is for images ONLY; map is unchanged.", fill="black", font=font)
    if cfg is None:
        text = "Height variation is vertical at each station. Exact view angles/heights are stored per frame."
    else:
        text = f"Map buffer: {required_clearance(cfg):.2f} m | red tint: no spawn | amber tint: auto buffer near unknown."
    d.text((8, 46), text, fill="black", font=font)
    skipped = (diagnostics or {}).get("rejected_count", 0)
    d.text((8, 66), f"Red X: skipped selected sample ({skipped}). Gray permission does NOT bypass wall clearance.", fill="black", font=font)
    def pix(xy):
        x, y = world_to_pixel(fmap, xy, base.size)
        return x, y + HEADER
    for corridor in selection.get("corridors", []):
        pts = [pix(p) for p in corridor["points"]]
        if len(pts) > 1:
            d.line(pts, fill="#44dfd3", width=3)
        for x, y in pts:
            d.rectangle((x-3, y-3, x+3, y+3), outline="#ffffff", width=1)
    px_scale = base.width / (fmap.shape[1] * fmap.resolution)
    for disk in selection.get("excluded_capture_disks", []):
        x, y = pix([disk["x"], disk["y"]]); r = disk["radius"] * px_scale
        d.ellipse((x-r, y-r, x+r, y+r), outline="#ff7777", width=2)
    for rejected in (diagnostics or {}).get("rejected_explicit_samples", []):
        x, y = pix([rejected["x"], rejected["y"]])
        d.line((x-4, y-4, x+4, y+4), fill="#ff4433", width=2)
        d.line((x-4, y+4, x+4, y-4), fill="#ff4433", width=2)
    for i, s in enumerate(stations):
        x, y = pix([s["x"], s["y"]])
        color = ("#c281f4" if s["unknown_capture_only"] else
                 "#e7ae32" if s["selection"] == "manual_pin" else
                 "#42d1c8" if s.get("corridor_id") else "#2476da")
        radius = 4 if s["station_id"] in pilots else 3
        d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="white", width=1)
        if all_labels or s["station_id"] in pilots or i % 20 == 0:
            label = s["station_id"] + (" P" if s["station_id"] in pilots else "")
            box = d.textbbox((0, 0), label, font=font, stroke_width=1)
            label_w = box[2] - box[0]
            label_x = x + 6 if x + 6 + label_w < image.width - 2 else x - label_w - 6
            label_x = max(2, label_x)
            d.text((label_x, max(HEADER + 2, y-7)), label, fill="white", font=font,
                   stroke_width=1, stroke_fill="#202020")
    return image


def save_dense_plan(out, plan, selection):
    import csv
    out = Path(out)
    base_save_plan(out, plan)
    fmap = FloorMap.load(plan["inputs"]["map"]["yaml"])
    write_json(out / "selection.json", {"map_identity": plan["inputs"]["map"], **selection})
    preview(fmap, plan["stations"], selection, plan["pilot_station_ids"],
            cfg=plan["config"], diagnostics=plan.get("spawn_clearance")).save(out / "camera_positions.png")
    preview(fmap, plan["stations"], selection, plan["pilot_station_ids"], all_labels=True,
            max_size=(1700, 1300), cfg=plan["config"], diagnostics=plan.get("spawn_clearance")).save(out / "camera_positions_labels.png")
    if "spawn_clearance" in plan:
        write_json(out / "spawn_clearance_report.json", plan["spawn_clearance"])
    frames = plan["frames"]
    with (out / "frame_poses.csv").open("w", newline="") as f:
        keys = ["frame_id", "station_id", "x", "y", "camera_height_m", "yaw_deg", "pitch_deg", "pilot", "unknown_capture_only", "corridor_id",
                "spawn_wall_clearance_lower_bound_m", "spawn_required_clearance_m"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(frames)
    heights = np.array([f["camera_height_m"] for f in frames])
    write_json(out / "sampling_summary.json", {
        "generator": plan["generator"], "seed": plan["config"]["seed"],
        "requested_positions": plan["config"]["requested_positions"], "actual_positions": len(plan["stations"]),
        "spawn_clearance_m": required_clearance(plan["config"]),
        "skipped_explicit_samples": plan.get("spawn_clearance", {}).get("rejected_count", 0),
        "planned_images": len(frames), "pilot_images": sum(f["pilot"] for f in frames),
        "corridor_positions": sum(bool(s.get("corridor_id")) for s in plan["stations"]),
        "gray_capture_only_positions": sum(s["unknown_capture_only"] for s in plan["stations"]),
        "actual_height_min_max_m": [float(heights.min()), float(heights.max())],
        "height_bins_m": np.linspace(*plan["config"]["height_range_m"], 6).tolist(),
        "counts_in_height_bins": np.histogram(heights, bins=np.linspace(*plan["config"]["height_range_m"], 6))[0].tolist(),
        "raw_float_array_gib": len(frames) * plan["config"]["width"] * plan["config"]["height"] * 12 / 1024**3,
        "note": "Counts cover planned images, not necessarily usable object views. No map/3-D collision validation."})
