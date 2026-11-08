from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .core import FloorMap, check_identity, empty_region, generate, import_selection, save_results


def main(argv=None):
    p=argparse.ArgumentParser(description='Camera spawns only: dark blocked, gray allowed in a reviewed outline, soft 18 cm clearance.')
    p.add_argument('--map',default='outputs/ros/valmet_map.yaml')
    p.add_argument('--out',default='outputs/camera_spawns_soft_v1')
    p.add_argument('--positions',type=int,default=160)
    p.add_argument('--preferred-clearance',type=float,default=.18,help='Soft target, provisional metres; never a cutoff.')
    p.add_argument('--max-shift',type=float,default=.30,help='Maximum local relocation path, in provisional metres.')
    p.add_argument('--corridor-spacing',type=float,default=.35,help='Seed step on optional priority lines.')
    p.add_argument('--gray-fraction',type=float,default=.20,help='Desired seed share from gray patches; not a promise of coverage.')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scale',type=float,default=1.43,help='Existing scene-to-map scale; does not rescale YAML.')
    p.add_argument('--floor-y',type=float,default=1.1)
    p.add_argument('--camera-height',type=float,default=1.7,help='Only for preview GS XYZ; no views rendered.')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--selection',help='Import earlier dense selection pins/corridors. Draw room outline once.')
    group.add_argument('--region',help='Reuse a saved camera_spawn_soft region.json including room outline.')
    p.add_argument('--headless',action='store_true',help='Requires --region with a completed outline.')
    a=p.parse_args(argv)
    try:
        out=Path(a.out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Use a new, empty --out directory.')
        fmap=FloorMap.load(a.map)
        region=empty_region()
        if a.selection:
            region=import_selection(a.selection,fmap)
        if a.region:
            saved=json.loads(Path(a.region).read_text());check_identity(saved,fmap)
            region={k:saved.get(k,[]) for k in region}
        if a.headless:
            if not region['polygon']:
                raise ValueError('--headless requires a saved --region with an approved room outline.')
            records,report,layers=generate(fmap,region,count=a.positions,preferred=a.preferred_clearance,
                max_shift=a.max_shift,corridor_spacing=a.corridor_spacing,gray_fraction=a.gray_fraction,
                seed=a.seed,scale=a.scale,floor_y=a.floor_y,camera_height=a.camera_height)
            save_results(a.out,fmap,region,records,report,layers)
            print(f"Saved {len(records)} positions to {a.out}; "
                  f"{report['below_preferred_clearance']} below preferred clearance intentionally retained.")
        else:
            import tkinter as tk
            from .editor import Editor
            root=tk.Tk();Editor(root,fmap,region,a);root.mainloop()
    except KeyboardInterrupt:
        print('Interrupted; source map and existing outputs unchanged.',file=sys.stderr);return 130
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
