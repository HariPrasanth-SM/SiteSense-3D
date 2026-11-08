from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .common import read_json,disjoint_output,lock


def make_parser():
    p=argparse.ArgumentParser(description='Stage 3: DINOv3 features for all masks; keep-only association candidates.')
    sub=p.add_subparsers(dest='command',required=True)
    def source_flags(q):
        q.add_argument('--dataset',default='outputs/datasets/valmet_views_soft200_v1')
        q.add_argument('--segmentation',default='outputs/segmentation/valmet_soft200_fastsam_v1')
        q.add_argument('--out',required=True,type=Path)
    def model_flags(q):
        q.add_argument('--repo',default='vendor/dinov3')
        q.add_argument('--weights',required=True)
        q.add_argument('--arch',choices=['dinov3_vits16','dinov3_vits16plus','dinov3_vitb16','dinov3_vitl16'],default='dinov3_vitb16')
        q.add_argument('--device',default='cuda:0')
        q.add_argument('--precision',choices=['bf16','fp16','fp32'],default='bf16')
    a=sub.add_parser('audit',help='Verify source contracts and counts; no torch/model required.');source_flags(a)
    c=sub.add_parser('check-model',help='Strict weight load and one small native forward pass.');model_flags(c)
    e=sub.add_parser('extract',help='Archive descriptors; use --pilot first.');source_flags(e);model_flags(e)
    e.add_argument('--crop-size',type=int,default=256)
    e.add_argument('--frame-side',type=int,default=640)
    e.add_argument('--crop-pad',type=float,default=.08)
    e.add_argument('--batch-size',type=int,default=16)
    e.add_argument('--no-dense',action='store_true',help='Omit dense per-frame token cache (new run only).')
    e.add_argument('--pilot',action='store_true');e.add_argument('--pilot-frames',type=int,default=32)
    e.add_argument('--limit',type=int,help='Maximum NEW frame archives to compute.')
    for cmd in ('report','reindex'):
        q=sub.add_parser(cmd,help='Rebuild keep-only lists/review from completed features without a model.')
        q.add_argument('--out',type=Path,required=True)
    return p


def main(argv=None):
    p=make_parser();a=p.parse_args(argv)
    try:
        if a.command=='check-model':
            import numpy as np
            from .model import DINOEncoder
            model=DINOEncoder(a.repo,a.weights,a.arch,a.device,a.precision)
            tokens,cls=model.infer(np.zeros((1,3,256,256),np.float32))
            print('Pretrained model forward PASS:',tokens.shape,cls.shape)
            print('This validates loading/API only, not retrieval accuracy.');return
        from .source import Source,audit
        from .pipeline import extract,build_index
        from .report import make_report
        if a.command in ('audit','extract'):
            if a.command=='extract':
                if a.batch_size<1 or a.pilot_frames<1 or (a.limit is not None and a.limit<1): raise ValueError('Counts must be positive')
                if a.crop_size%16 or a.frame_side%16 or min(a.crop_size,a.frame_side)<16: raise ValueError('Input sides must be multiples of 16')
                if not 0<=a.crop_pad<=1: raise ValueError('crop-pad must be between 0 and 1')
            src=Source(a.dataset,a.segmentation);out=disjoint_output(a.out,src.dataset,src.seg)
            with lock(out):
                if a.command=='audit':audit(src,out)
                else:
                    extract(src,out,a)
                    make_report(out)
        else:
            out=a.out.expanduser().resolve();bind=read_json(out/'source.json');src=Source(bind['dataset'],bind['segmentation'])
            if src.binding!=bind: raise ValueError('Source identity changed')
            with lock(out):
                build_index(src,out)
                make_report(out)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed frame features are retained; rerun the identical extraction command.',file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'\nSTAGE 3 ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        if 'out of memory' in str(exc).lower():
            print('Reduce --batch-size to 4 (same feature recipe, safe to resume). Close other GPU jobs. Do not reinstall CUDA.',file=sys.stderr)
        raise


if __name__=='__main__':main()
