from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from .common import read_json, lock, disjoint, write_json
from .catalog import Inputs, Catalog, prepare
from .geometry import MatchPolicy
from .pipeline import run_queries
from .report import make_report


def parser():
    p=argparse.ArgumentParser(description='Valmet Stage 4A: kept-mask geometry-verified association pilot; no model inference')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare',help='Build compact keep-only geometry/feature catalog')
    a.add_argument('--dataset',required=True);a.add_argument('--segmentation',required=True)
    a.add_argument('--features',required=True);a.add_argument('--out',required=True)
    for command in ('pilot','inspect'):
        a=sub.add_parser(command,help='Compare seed masks against other stations; no transitive merge')
        a.add_argument('--out',required=True,help='Prepared catalog output')
        a.add_argument('--name',default='pilot_v1' if command=='pilot' else 'inspection_v1',help='New name for changed thresholds/queries')
        if command=='pilot':a.add_argument('--queries',type=int,default=24)
        else:
            a.add_argument('--frame',required=True);a.add_argument('--mask',type=int,required=True)
        a.add_argument('--global-candidates',type=int,default=24);a.add_argument('--local-candidates',type=int,default=24)
        a.add_argument('--policy',type=Path,help='Optional JSON MatchPolicy overrides; recorded in run.json')
    a=sub.add_parser('report');a.add_argument('--out',required=True);a.add_argument('--name',default='pilot_v1')
    return p


def main():
    args=parser().parse_args()
    try:
        out=Path(args.out).expanduser().resolve()
        if args.command=='prepare':
            inp=Inputs(args.dataset,args.segmentation,args.features)
            out=disjoint(out,inp.dataset,inp.seg,inp.feat)
            with lock(out):prepare(inp,out)
        else:
            if not args.name or Path(args.name).name!=args.name or args.name in ('.','..'):
                raise ValueError('--name must be a simple folder name')
            with lock(out):
                catalog=Catalog(out)
                runout=out/'runs'/args.name
                if args.command=='report':make_report(catalog,runout);return
                p=MatchPolicy(**(read_json(args.policy) if args.policy else {})).validate()
                if not 0<=args.global_candidates<=500 or not 0<=args.local_candidates<=500 or not (args.global_candidates+args.local_candidates):
                    raise ValueError('Candidate counts must be 0..500, with at least one positive count.')
                if args.command=='pilot':
                    if not 1<=args.queries<=200:raise ValueError('Start with 1..200 seeds; full-scene clustering is not this module.')
                    queries=catalog.seed_indices(args.queries)
                else:
                    key=(args.frame,args.mask)
                    if key not in catalog.lookup:
                        raise ValueError('Requested mask is not a strict keep+geometry candidate. Do not bypass QA selection.')
                    queries=[catalog.lookup[key]]
                runout.mkdir(parents=True,exist_ok=True)
                run_queries(catalog,queries,p,runout,args.global_candidates,args.local_candidates)
                make_report(catalog,runout)
    except KeyboardInterrupt:
        print('\nStopped; completed query results remain. Re-run the identical command to resume.',file=sys.stderr)
        return 130
    except (ValueError,OSError,KeyError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':sys.exit(main())
