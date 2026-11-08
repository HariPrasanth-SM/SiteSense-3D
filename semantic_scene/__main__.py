from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .common import lock


def parser():
    p=argparse.ArgumentParser(description='Valmet 3D region layer + editable labels; no navigation changes')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('build');a.add_argument('--association',required=True);a.add_argument('--run',default='pilot_v1');a.add_argument('--out',required=True)
    a.add_argument('--samples',type=int,default=12000);a.add_argument('--max-views',type=int,default=9)
    a.add_argument('--min-views',type=int,default=3);a.add_argument('--voxel',type=float,default=.02)
    a=sub.add_parser('view');a.add_argument('--scene',required=True);a.add_argument('--environment',type=Path)
    a.add_argument('--max-env-points',type=int,default=200000);a.add_argument('--max-region-points',type=int,default=15000)
    a.add_argument('--port',type=int,default=8765);a.add_argument('--open-browser',action='store_true')
    a=sub.add_parser('suggest-labels');a.add_argument('--scene',required=True);a.add_argument('--categories',type=Path)
    a.add_argument('--model',default='ViT-B-32');a.add_argument('--pretrained',default='laion2b_s34b_b79k');a.add_argument('--device',default='cuda:0')
    a.add_argument('--views',type=int,default=3);a.add_argument('--mask-weight',type=float,default=.7)
    a.add_argument('--min-score',type=float,default=.22);a.add_argument('--min-margin',type=float,default=.025);a.add_argument('--min-votes',type=float,default=.67)
    a=sub.add_parser('check');a.add_argument('--scene',required=True)
    return p


def main():
    args=parser().parse_args()
    try:
        if args.command=='build':
            if not 32<=args.samples<=20000 or not 2<=args.max_views<=32 or not 2<=args.min_views<=args.max_views or not .001<=args.voxel<=.2:
                raise ValueError('Invalid samples/views/voxel settings')
            if Path(args.run).name!=args.run or args.run in ('.','..'):raise ValueError('Use a simple run folder name')
            # Reject overlapping output paths BEFORE opening a lock or creating a file there.
            from .common import read_json
            src=read_json(Path(args.association)/'catalog_complete.json')['config']['source']
            target=Path(args.out).expanduser().resolve()
            for root in [Path(args.association).expanduser().resolve(),*[Path(src[k]).resolve() for k in ('dataset','segmentation','features')]]:
                if target==root or root in target.parents or target in root.parents:
                    raise ValueError('Output must be separate from all source directories')
            from .build import build
            with lock(args.out):build(args)
        elif args.command=='suggest-labels':
            if not 2<=args.views<=16 or not 0<=args.min_score<=1 or not 0<=args.min_margin<=2 or not 0<=args.min_votes<=1:
                raise ValueError('Invalid labeling parameters')
            from .semantics import suggest
            with lock(args.scene):suggest(args)
        elif args.command=='check':
            from .common import load_scene,read_json,checked
            out,scene=load_scene(args.scene)
            for r in scene['regions']:
                folder=out/'regions'/r['region_id'];stamp=read_json(folder/'complete.json')
                if stamp['scene_id']!=scene['scene_id']:raise ValueError('Region scene ID mismatch')
                for n,e in stamp['files'].items():checked(folder/n,e)
            print(f'PASS: {len(scene["regions"])} regions; independent files and checksums verified.')
        else:
            from .viewer import serve
            serve(args)
    except KeyboardInterrupt:
        print('\nStopped. Completed region files are preserved.');return 130
    except (ValueError,OSError,KeyError,RuntimeError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    return 0

if __name__=='__main__':sys.exit(main())
