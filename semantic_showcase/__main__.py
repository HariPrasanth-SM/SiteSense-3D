from __future__ import annotations
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Add a hand-picked rotating 2-row finale to your existing successful animation. Originals unchanged.')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('pick',help='Local image/point-cloud picker; save ordered region IDs. No GPU.')
    a.add_argument('--base',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--port',type=int,default=8766);a.add_argument('--open-browser',action='store_true')
    a=s.add_parser('prepare',help='Keep/retime finished intro and prepare selected two-row camera/object motion.')
    a.add_argument('--base',type=Path,required=True);a.add_argument('--selection',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True);a.add_argument('--duration',type=float,default=25.)
    a.add_argument('--intro-duration',type=float,help='Optional explicit retiming of the original intro. Default retains its exact duration.')
    a.add_argument('--formation',type=float,default=1.4);a.add_argument('--pullback',type=float,default=1.4)
    a.add_argument('--rotation-deg',type=float,default=35.,help='Slow total turn during finale; adjacent rows turn in opposite directions.')
    a.add_argument('--gap',type=float,default=.55);a.add_argument('--forward-distance',type=float)
    a.add_argument('--hero-scale',type=float,default=1.,help='Uniform scale for ALL chosen regions, recorded. Default no resizing.')
    a=s.add_parser('render',help='Copy validated intro PNGs; render only new finale with gsplat.')
    a.add_argument('--out',type=Path,required=True);a.add_argument('--preview',action='store_true');a.add_argument('--limit',type=int)
    a=s.add_parser('encode',help='Combine all frames into a 25s (or chosen duration) MP4.')
    a.add_argument('--out',type=Path,required=True);a.add_argument('--crf',type=int,default=18)
    g=a.add_mutually_exclusive_group();g.add_argument('--clean',action='store_true');g.add_argument('--both',action='store_true')
    args=p.parse_args()
    if args.command=='pick':
        if not 1024<=args.port<=65535:p.error('Use port 1024–65535')
        from .picker import serve
        serve(args)
    elif args.command=='prepare':
        if not 10<=args.duration<=120 or args.formation<=0 or args.pullback<=0 or not 0<=args.rotation_deg<=360 or args.gap<0 or args.hero_scale<=0:
            p.error('Invalid duration/formation/pullback/rotation/gap/scale')
        if args.forward_distance is not None and args.forward_distance<0:p.error('forward-distance must be nonnegative')
        from .timeline import prepare
        prepare(args)
    elif args.command=='render':
        if args.limit is not None and args.limit<1:p.error('limit must be positive')
        from .render import render
        render(args)
    else:
        if not 0<=args.crf<=40:p.error('Use crf 0–40')
        from .render import encode
        encode(args)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nStopped. Completed files remain available.')
