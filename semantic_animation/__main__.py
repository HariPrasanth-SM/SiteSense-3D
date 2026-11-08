from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Existing room splats -> point cloud -> exploded regions -> original region splats. No training.')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='Read existing exports, copy assets, create layout and explicit camera/object timeline (CPU).')
    a.add_argument('--enhancement',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--environment',type=Path,help='Existing XYZRGB NPZ in map coordinates. Omit to derive centers/DC from room PLY.')
    a.add_argument('--scene-ply',type=Path,help='Original whole-room Gaussian PLY; normally read from enhancement config.')
    a.add_argument('--camera',type=Path,help='Known good camera JSON with camera_to_map or original-GS camera_to_world.')
    a.add_argument('--objects',type=int,default=12,help='Automatic showreel region budget, not an object-count claim.')
    pick=a.add_mutually_exclusive_group()
    pick.add_argument('--regions',help='Explicit comma-separated region IDs; overrides automatic budget and structural/duplicate heuristics.')
    pick.add_argument('--all-regions',action='store_true',help='Include ALL prepared, nonempty Gaussian regions; no size/structure/overlap/count filter.')
    a.add_argument('--include-rejected',action='store_true',help='Only with --all-regions: explicitly include manually rejected regions too.')
    a.add_argument('--max-extent',type=float,default=5.0,help='Automatic selection max observed dimension, provisional map metres.')
    a.add_argument('--duration',type=float,default=18.)
    a.add_argument('--fps',type=int,default=24)
    a.add_argument('--width',type=int,default=1280)
    a.add_argument('--height',type=int,default=720)
    a.add_argument('--fov',type=float,default=55.,help='Vertical FOV in degrees.')
    a.add_argument('--columns',type=int,default=0,help='0 = automatic gallery column count.')
    a.add_argument('--gap',type=float,default=.70,help='Gallery slot gap in map units.')
    a.add_argument('--env-points',type=int,default=180000)
    a.add_argument('--object-points',type=int,default=14000,help='Display cap only; full cleaned clouds and real splats are preserved.')
    a.add_argument('--point-sigma',type=float,default=.65,help='Point-cloud dot sigma in pixels, not physical Gaussian scale.')
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--portable',action='store_true',help='Also copy the original room PLY into the animation assets.')
    r=sub.add_parser('render',help='CUDA rendering with persistent scene/asset loading and frame-level resume.')
    r.add_argument('--out',type=Path,required=True)
    r.add_argument('--preview',action='store_true',help='Ten spaced frames, reused by the later full render.')
    r.add_argument('--limit',type=int,help='At most this many NEW frames this run.')
    e=sub.add_parser('encode',help='Encode validated PNG frames with FFmpeg.')
    e.add_argument('--out',type=Path,required=True)
    e.add_argument('--crf',type=int,default=18)
    g=e.add_mutually_exclusive_group();g.add_argument('--clean',action='store_true');g.add_argument('--both',action='store_true')
    args=p.parse_args()
    if args.command=='prepare':
        if args.objects<1 or not 10<=args.fov<=100 or args.gap<0 or args.env_points<=0 or args.object_points<=0 or args.point_sigma<=0 or args.columns<0:
            p.error('Invalid object count, FOV, gap, display cap, point size, or column count.')
        if args.include_rejected and not args.all_regions:p.error('--include-rejected requires --all-regions')
        from .plan import prepare
        prepare(args)
    elif args.command=='render':
        if args.limit is not None and args.limit<1:p.error('--limit must be positive')
        from .movie import render
        render(args)
    else:
        if not 0<=args.crf<=40:p.error('Use CRF 0–40')
        from .movie import encode
        encode(args)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nInterrupted. Completed frames remain reusable.')
