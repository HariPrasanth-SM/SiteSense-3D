from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Additive object cleanup, Gaussian export and evidence viewer')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='CPU: filter region surfaces, overlay masks, optionally export original splats')
    a.add_argument('--scene',required=True,type=Path);a.add_argument('--out',required=True,type=Path)
    a.add_argument('--gaussian-ply',type=Path)
    a.add_argument('--neighbors',type=int,default=24);a.add_argument('--std-ratio',type=float,default=2.0)
    a.add_argument('--splat-radius',type=float,default=.08,help='Maximum map distance from cleaned visible surface, metres')
    a.add_argument('--min-splat-views',type=int,default=2)
    a.add_argument('--depth-abs',type=float,default=.12);a.add_argument('--depth-rel',type=float,default=.03)
    a.add_argument('--max-gaussian-sigma',type=float,default=0.,help='Optional max activated sigma in map metres; 0 disables size filter')
    a.add_argument('--limit',type=int);a.add_argument('--region',help='Optional exact region ID')
    a=sub.add_parser('render-previews',help='GPU: render each subset at original source cameras')
    a.add_argument('--enhancement',required=True,type=Path);a.add_argument('--views',type=int,default=3)
    a.add_argument('--width',type=int,default=640);a.add_argument('--limit',type=int);a.add_argument('--region')
    a=sub.add_parser('view',help='Local browser with clean/raw cloud, splat centers, and optional live splats')
    a.add_argument('--enhancement',required=True,type=Path);a.add_argument('--environment',type=Path)
    a.add_argument('--max-env-points',type=int,default=200000);a.add_argument('--max-region-points',type=int,default=15000)
    a.add_argument('--live-splats',action='store_true',help='Enable on-demand GPU render of selected region from browser view')
    a.add_argument('--port',type=int,default=8765);a.add_argument('--open-browser',action='store_true')
    args=p.parse_args()
    try:
        if args.command=='prepare':
            import math
            if args.neighbors<2 or not math.isfinite(args.std_ratio) or args.std_ratio<=0:raise ValueError('Invalid SOR settings')
            if not 0<args.splat_radius<=1 or not 1<=args.min_splat_views<=32:raise ValueError('Invalid splat distance/view settings')
            if not 0<args.depth_abs<=2 or not 0<=args.depth_rel<=1 or not 0<=args.max_gaussian_sigma<=5:raise ValueError('Invalid depth/size settings')
            from .prepare import prepare
            prepare(args)
        elif args.command=='render-previews':
            from .splats import render_previews
            render_previews(args)
        else:
            from .viewer import serve
            serve(args)
    except KeyboardInterrupt:
        print('\nStopped. Completed outputs are preserved.');return 130
    except (ValueError,RuntimeError,OSError,KeyError,ImportError) as e:
        print(f'ERROR: {e}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())
