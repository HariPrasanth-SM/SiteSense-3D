from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import (FloorMap, create_plan, read_json, save_plan, select_stations)


def planning_args(p):
    p.add_argument("--map", default="outputs/ros/valmet_map.yaml")
    p.add_argument("--scene", default="data/Industrial Boiler Room.ply")
    p.add_argument("--out", default="outputs/datasets/valmet_views_v1")
    p.add_argument("--positions", type=int, default=40, help="Total camera stations, including explicit pins.")
    p.add_argument("--clearance", type=float, default=0.10, help="Auto-camera minimum map clearance in provisional m; NOT avatar radius.")
    p.add_argument("--spacing", type=float, default=0.50, help="Minimum spacing for automatically added stations, in provisional m.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-height", type=float, default=1.7)
    p.add_argument("--scale", type=float, default=1.43)
    p.add_argument("--floor-y", type=float, default=1.1)
    p.add_argument("--fov", type=float, default=70., help="Vertical field of view in degrees.")
    p.add_argument("--yaw-step", type=int, default=60, help="Must divide 360 exactly.")
    p.add_argument("--pitches", type=float, nargs="+", default=[0., -20.])
    p.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--alpha-min", type=float, default=0.2, help="Validity-mask threshold; raw depth/alpha retained.")


def config_from_args(a):
    import math
    numeric = [a.scale, a.floor_y, a.camera_height, a.fov, a.clearance, a.spacing, a.alpha_min] + a.pitches
    if not all(math.isfinite(v) for v in numeric): raise ValueError("All numeric options must be finite.")
    if a.width < 32 or a.height < 32 or a.width > 4096 or a.height > 4096:
        raise ValueError("Image dimensions must be between 32 and 4096.")
    if a.scale <= 0 or a.camera_height <= 0 or not 1 < a.fov < 179:
        raise ValueError("Invalid scene scale, camera height or FOV.")
    if a.yaw_step < 1 or a.yaw_step > 360 or 360 % a.yaw_step:
        raise ValueError("--yaw-step must be a positive divisor of 360.")
    if not a.pitches or any(abs(p) >= 85 for p in a.pitches):
        raise ValueError("Camera pitches must be between -85 and +85 degrees.")
    if len(set(a.pitches)) != len(a.pitches) or not 0 <= a.alpha_min <= 1:
        raise ValueError("Duplicate pitches or invalid alpha cutoff.")
    return {"width": a.width, "height": a.height, "camera_height_m": a.camera_height,
            "scale": a.scale, "floor_y_scene": a.floor_y, "fov_y_deg": a.fov,
            "yaws_deg": list(range(0, 360, a.yaw_step)), "pitches_deg": a.pitches,
            "sh_degree": a.sh_degree, "alpha_min": a.alpha_min, "near_m": 0.03, "far_m": 60.,
            "requested_positions": a.positions, "auto_clearance_m": a.clearance,
            "auto_spacing_m": a.spacing, "pixel_center_offset": [0.5, 0.5],
            "renderer_mode": "RGB+ED", "scale_is_provisional": True}


def main():
    parser = argparse.ArgumentParser(description="Stage 1: calibrated synthetic views; no map edits, training or segmentation.")
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Click capture-only pins and save the immutable camera plan.")
    planning_args(select)
    plan = sub.add_parser("plan", help="Headless camera plan, optionally from a saved selection JSON.")
    planning_args(plan)
    plan.add_argument("--selection", help="Previously saved selection.json (must refer to the same map).")
    render = sub.add_parser("render", help="Render/resume RGB, raw depth, alpha and exact camera matrices.")
    render.add_argument("--out", default="outputs/datasets/valmet_views_v1")
    render.add_argument("--pilot", action="store_true", help="Only render the spatially spread pilot views (usually 36).")
    render.add_argument("--limit", type=int, help="Maximum new frames THIS invocation; useful for a first-frame check.")
    report = sub.add_parser("report", help="Verify frames and regenerate quality table/contact sheets.")
    report.add_argument("--out", default="outputs/datasets/valmet_views_v1")
    args = parser.parse_args()
    try:
        if args.command in ("select", "plan"):
            cfg = config_from_args(args)
            if (Path(args.out) / "plan.json").exists():
                raise FileExistsError("Dataset already has a plan. Use a new --out to change sampling/settings; render resumes an existing plan.")
            if args.command == "select":
                from .picker import run_picker
                run_picker(args, cfg)
            else:
                from .common import map_identity
                pins, excluded = [], []
                if args.selection:
                    s = read_json(args.selection)
                    if s["map_identity"] != map_identity(args.map):
                        raise ValueError("Selection belongs to a different map.")
                    pins, excluded = s["pins"], s.get("excluded_capture_disks", [])
                fmap = FloorMap.load(args.map)
                stations = select_stations(fmap, pins, excluded, args.positions, args.clearance, args.spacing)
                p = create_plan(args.map, args.scene, stations, pins, excluded, cfg)
                save_plan(args.out, p)
                print(f"Saved {len(stations)} positions and {len(p['frames'])} views. Pilot {sum(v['pilot'] for v in p['frames'])} views.")
                if len(stations) < args.positions:
                    print("Fewer positions met the configured spacing/clearance. Review plan, or create another dataset with different settings.")
        elif args.command == "render":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive.")
            from .capture import render
            render(args.out, args.pilot, args.limit)
        else:
            from .capture import report
            report(args.out)
    except KeyboardInterrupt:
        print("Interrupted; no source map or scene was modified.", file=sys.stderr)
        return 130
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {exc}\nNo CUDA/package changes or input map edits were attempted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
