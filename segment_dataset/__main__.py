from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from .quality import Policy
from .pipeline import setup,audit,segment,reselect
from .report import build_report
from .storage import output_lock,read_json,unpack_masks,write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Stage 2: FastSAM archive + reversible RGB/depth quality selection')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('audit','segment','reselect','report'):
        ap=sub.add_parser(name)
        ap.add_argument('--dataset',type=Path,required=True)
        ap.add_argument('--out',type=Path,required=True)
        ap.add_argument('--policy',type=Path,help='JSON threshold overrides; omission reuses output quality_policy.json')
        if name=='segment':
            ap.add_argument('--model',default='FastSAM-s.pt')
            ap.add_argument('--device',default='0')
            ap.add_argument('--imgsz',type=int,default=640)
            ap.add_argument('--conf',type=float,default=0.25)
            ap.add_argument('--iou',type=float,default=0.9)
            ap.add_argument('--max-det',type=int,default=300)
            ap.add_argument('--pilot',action='store_true')
            ap.add_argument('--pilot-frames',type=int,default=48)
            ap.add_argument('--limit',type=int,help='Maximum NEW segmentation frames; restart resumes completed archives')
            ap.add_argument('--segment-excluded',action='store_true',help='Archive proposals for readable severe-QA frames too; never overrides corrupt inputs')
    ep=sub.add_parser('export-mask'); ep.add_argument('--out',type=Path,required=True)
    ep.add_argument('--frame',required=True); ep.add_argument('--mask',type=int,required=True)
    ep.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='export-mask':
        if args.dest.exists(): raise FileExistsError('Choose a new export folder')
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',args.frame): raise ValueError('Bad frame ID')
        source=read_json(args.out/'source.json'); masks,scores,boxes=unpack_masks(args.out/'frames'/args.frame/'masks.npz')
        if not 0<=args.mask<len(masks): raise ValueError(f'Mask index must be 0..{len(masks)-1}')
        rgb=Image.open(Path(source['dataset_path'])/'frames'/args.frame/'rgb.png').convert('RGB')
        arr=np.array(rgb); mask=masks[args.mask]
        args.dest.mkdir(parents=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(args.dest/'mask.png')
        rgba=np.dstack([arr,mask.astype(np.uint8)*255]); Image.fromarray(rgba).save(args.dest/'masked_rgba.png')
        yy,xx=np.nonzero(mask)
        if len(xx): Image.fromarray(rgba[yy.min():yy.max()+1,xx.min():xx.max()+1]).save(args.dest/'crop_rgba.png')
        write_json(args.dest/'observation.json',{'frame_id':args.frame,'mask_index':args.mask,'score':float(scores[args.mask]),'model_bbox_xyxy':boxes[args.mask].tolist(),
                    'source_dataset':source['dataset_path'],'plan_hash':source['plan_hash']})
        print('Exported raw original-pixel mask to',args.dest); return
    if args.command=='segment':
        if args.imgsz<32 or not 0<=args.conf<=1 or not 0<=args.iou<=1 or args.max_det<1:
            parser.error('Invalid inference settings')
        if args.pilot_frames<1 or (args.limit is not None and args.limit<1): parser.error('Counts must be positive')
    # Do not create a lock/output inside the source capture directory.
    ds=args.dataset.resolve(); out=args.out.resolve()
    if out==ds or ds in out.parents or out in ds.parents: parser.error('Output must be separate from source dataset')
    with output_lock(out):
        dataset,out,plan,p,ph=setup(ds,out,args.policy)
        if args.command=='report':
            records=[json.loads(l) for l in (out/'frame_audit.jsonl').read_text().splitlines() if l]
            if any(r['policy_hash']!=ph for r in records): raise ValueError('Policy changed: run audit or reselect, not report')
        else:
            records=audit(dataset,out,plan,p,ph)
        try:
            if args.command=='segment': segment(dataset,out,plan,records,p,ph,args)
            if args.command=='reselect': reselect(dataset,out,plan,records,p,ph)
        except KeyboardInterrupt:
            print('\nInterrupted; complete archives preserved. Re-run identical segment command to resume.',flush=True)
        finally:
            build_report(dataset,out,plan,records,ph)

if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,FileNotFoundError,KeyError,ImportError,OSError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise
