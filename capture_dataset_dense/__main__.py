from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from .planning import (FloorMap, PITCH_POLICY, choose_stations, create_dense_plan,
                       load_selection, save_dense_plan)


def planning_args(p):
    p.add_argument('--map', default='outputs/ros/valmet_map.yaml')
    p.add_argument('--scene', default='data/Industrial Boiler Room.ply')
    p.add_argument('--out', default='outputs/datasets/valmet_views_2000_v1')
    p.add_argument('--selection', help='Reuse old or dense selection.json, then optionally add corridor lines.')
    p.add_argument('--positions', type=int, default=200, help='Total horizontal stations, including pins/corridors.')
    p.add_argument('--frames', type=int, default=2000, help='Exact frame budget, split evenly over actual stations.')
    p.add_argument('--height-min', type=float, default=.5, help='Minimum camera height in provisional metres.')
    p.add_argument('--height-max', type=float, default=2.5, help='Maximum camera height in provisional metres.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--spacing', type=float, default=.25, help='Minimum distance for automatic stations.')
    p.add_argument('--clearance', '--wall-clearance', dest='clearance', type=float, default=.25,
                   help='Minimum 2-D map-edge buffer for ALL camera spawns, in provisional metres (default 0.25).')
    p.add_argument('--corridor-spacing', type=float, default=.25, help='Sampling step along user-drawn corridor; vertices snap to map cells.')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480, help='Image height in pixels, NOT camera elevation.')
    p.add_argument('--scale', type=float, default=1.43)
    p.add_argument('--floor-y', type=float, default=1.1)
    p.add_argument('--fov', type=float, default=70., help='Vertical field of view, degrees.')
    p.add_argument('--sh-degree', type=int, choices=[0,1,2,3], default=3)
    p.add_argument('--alpha-min', type=float, default=.2)
    p.add_argument('--pilot-positions', type=int, default=8)
    p.add_argument('--pilot-views', type=int, default=6, help='Views per pilot station, distributed over low/mid/high elevations.')


def config_from_args(a):
    vals = [a.height_min, a.height_max, a.spacing, a.clearance, a.corridor_spacing,
            a.scale, a.floor_y, a.fov, a.alpha_min]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('Numeric settings must be finite.')
    if not 1 <= a.positions <= 500 or not a.positions <= a.frames <= 50000:
        raise ValueError('Use 1..500 positions and positions <= frames <= 50000.')
    if not 0 <= a.seed < 2**32:
        raise ValueError('Seed must be an integer between 0 and 2**32-1.')
    if not 0 < a.height_min < a.height_max <= 10:
        raise ValueError('Expected 0 < height-min < height-max <= 10 (provisional metres).')
    if not 32 <= a.width <= 4096 or not 32 <= a.height <= 4096:
        raise ValueError('Image dimensions must be 32..4096 pixels.')
    if a.spacing <= 0 or a.clearance < 0 or a.corridor_spacing <= 0:
        raise ValueError('Spacing must be positive; clearance must be nonnegative.')
    if a.scale <= 0 or not 1 < a.fov < 179 or not 0 <= a.alpha_min <= 1:
        raise ValueError('Invalid scale/FOV/alpha threshold.')
    if not 1 <= a.pilot_positions <= 50 or not 1 <= a.pilot_views <= 50:
        raise ValueError('Pilot station/view counts must be 1..50.')
    return {'width': a.width, 'height': a.height,
            'scale': a.scale, 'floor_y_scene': a.floor_y, 'fov_y_deg': a.fov,
            'sh_degree': a.sh_degree, 'alpha_min': a.alpha_min,
            'near_m': .03, 'far_m': 60., 'pixel_center_offset': [.5,.5],
            'renderer_mode': 'RGB+ED', 'scale_is_provisional': True,
            'requested_positions': a.positions, 'requested_frames': a.frames,
            'height_range_m': [a.height_min, a.height_max], 'seed': a.seed,
            'view_sampling': 'per-station stratified heights and full-turn yaw; independent seeded jitter',
            'pitch_policy': PITCH_POLICY, 'auto_spacing_m': a.spacing,
            'auto_clearance_m': a.clearance, 'clearance_applies_to': 'all_stations',
            'clearance_measure': 'conservative distance to occupied-cell/map-exterior boundary; auto also to unknown',
            'corridor_spacing_m': a.corridor_spacing,
            'pilot_positions': a.pilot_positions, 'pilot_views_per_position': a.pilot_views}


def print_summary(plan):
    ss, vv = plan['stations'], plan['frames']
    print(f"Saved {len(ss)} horizontal positions and {len(vv)} calibrated views.", flush=True)
    print(f"Corridor stations: {sum(bool(s.get('corridor_id')) for s in ss)}; "
          f"gray capture-only stations: {sum(s['unknown_capture_only'] for s in ss)}; "
          f"pilot: {sum(v['pilot'] for v in vv)} images.", flush=True)
    audit = plan.get('spawn_clearance', {})
    if audit:
        print(f"Camera buffer: {audit['required_clearance_m']:.2f} m; "
              f"minimum retained wall/map-edge lower bound: {audit['minimum_retained_wall_clearance_m']:.3f} m.", flush=True)
        print(f"Skipped {audit['rejected_count']} near-boundary pin/corridor samples (not relocated). "
              "Details: spawn_clearance_report.json", flush=True)
        if audit['rejected_count'] and plan.get('corridors') and not any(s.get('corridor_id') for s in ss):
            print('WARNING: No corridor samples survived. Review the corridor or explicitly choose a smaller buffer in a new plan.', flush=True)
    if len(ss) < plan['config']['requested_positions']:
        print('WARNING: fewer stations satisfy spacing/clearance. The frame budget is still met, '
              'but more views share each position. Review camera_positions.png.', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='Dense calibrated capture with explicit corridor sampling; no map or CUDA changes.')
    sub = p.add_subparsers(dest='command', required=True)
    for name, help_text in [('select', 'Open picker, draw corridors, and save a NEW plan.'),
                            ('plan', 'Headless plan from an existing selection, without opening a window.')]:
        planning_args(sub.add_parser(name, help=help_text))
    r = sub.add_parser('render', help='Render/resume using the unchanged Stage 1 Gaussian renderer.')
    r.add_argument('--out', default='outputs/datasets/valmet_views_2000_v1')
    r.add_argument('--pilot', action='store_true')
    r.add_argument('--limit', type=int, help='Maximum NEW frames this invocation; completed frames are reused.')
    q = sub.add_parser('report', help='Rebuild height-aware contact sheets and frame index without CUDA.')
    q.add_argument('--out', default='outputs/datasets/valmet_views_2000_v1')
    a = p.parse_args(argv)
    try:
        if a.command in ('plan', 'select'):
            cfg = config_from_args(a)
            if (Path(a.out) / 'plan.json').exists() or (Path(a.out) / 'frames').exists():
                raise FileExistsError('Use a NEW --out for changed poses/settings. Existing captures are not overwritten.')
            selection = load_selection(a.selection, a.map)
            if a.command == 'select':
                from .picker import run_picker
                run_picker(a, cfg, selection)
            else:
                fmap = FloorMap.load(a.map)
                stations = choose_stations(fmap, selection, cfg)
                plan = create_dense_plan(a.map, a.scene, stations, selection, cfg)
                save_dense_plan(a.out, plan, selection)
                print_summary(plan)
        elif a.command == 'render':
            if a.limit is not None and a.limit < 1:
                raise ValueError('--limit must be positive.')
            from capture_dataset.capture import render
            # This invokes the already-tested loader/rasterizer/checksum writer.
            render(a.out, a.pilot, a.limit)
            from .reports import add_dense_reports
            add_dense_reports(a.out)
        else:
            from capture_dataset.capture import report
            report(a.out)
            from .reports import add_dense_reports
            add_dense_reports(a.out)
    except KeyboardInterrupt:
        print('Interrupted. Source scene/map and completed frames are preserved.', file=sys.stderr)
        return 130
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'ERROR: {exc}\nNo package installation, source-map or CUDA edits were attempted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
