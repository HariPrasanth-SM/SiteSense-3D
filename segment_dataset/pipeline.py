"""Sequential FastSAM inference, independent archive, source verification and resumable QA."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import __version__
from .quality import Policy, frame_metrics, image_features, depth_support, mask_metrics, decide_mask
from .storage import (digest,sha,read_json,write_json,write_jsonl,load_plan,read_source,pack_masks,
                      unpack_masks,archive_valid,mark_complete)


def utc(): return datetime.now(timezone.utc).isoformat()

def setup(dataset,out,policy_path=None):
    dataset=Path(dataset).expanduser().resolve(); out=Path(out).expanduser().resolve()
    if out==dataset or dataset in out.parents or out in dataset.parents:
        raise ValueError('Use a separate Stage 2 output directory, not inside/above the source capture')
    plan=load_plan(dataset)
    out.mkdir(parents=True,exist_ok=True)
    expected={'schema':'valmet.stage2.source.v1','dataset_path':str(dataset),'plan_hash':plan['plan_hash'],
              'plan_file_sha256':sha(dataset/'plan.json')}
    if (out/'source.json').exists() and read_json(out/'source.json')!=expected:
        raise ValueError('Different source dataset at this output; choose a new output directory')
    write_json(out/'source.json',expected)
    if policy_path is None and (out/'quality_policy.json').exists(): policy_path=out/'quality_policy.json'
    policy=Policy.load(policy_path); ph=digest(asdict(policy))
    write_json(out/'quality_policy.json',asdict(policy)); write_json(out/'policies'/f'{ph}.json',asdict(policy))
    return dataset,out,plan,policy,ph


def audit(dataset,out,plan,p,ph):
    records=[]
    for i,v in enumerate(plan['frames'],1):
        rec={**v,'policy_hash':ph,'plan_hash':plan['plan_hash']}
        try:
            rgb,d,a,cam,prior,source_hash=read_source(dataset,v,plan,verify=True)
            rec.update(frame_metrics(rgb,d,a,p))
            rec.update(source_ok=True,source_complete_sha256=source_hash,
                       capture_flags=prior.get('flags',[]),height=rgb.shape[0],width=rgb.shape[1])
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            rec.update(source_ok=False,status='exclude',reasons=['source_or_calibration_invalid'],
                       warnings=[],capture_flags=[],error=f'{type(exc).__name__}: {exc}')
        write_json(out/'audit'/f"{v['frame_id']}.json",rec); records.append(rec)
        if i%100==0 or i==len(plan['frames']):
            print(f"QA [{i}/{len(plan['frames'])}] {dict(Counter(r['status'] for r in records))}",flush=True)
    write_jsonl(out/'frame_audit.jsonl',records)
    return records


def choose_pilot(records,count=48):
    """Capture pilots + flagged views + all height bands, never first-N only."""
    count=min(count,len(records)); chosen=[]; seen=set()
    def take(r):
        if r['frame_id'] not in seen and len(chosen)<count:
            seen.add(r['frame_id']); chosen.append(r)
    flagged=[r for r in records if r['status']!='keep' or r.get('capture_flags')]
    if flagged:
        for i in np.linspace(0,len(flagged)-1,min(len(flagged),max(1,count//3)),dtype=int): take(flagged[i])
    prior=[r for r in records if r.get('pilot')]
    if prior:
        for i in np.linspace(0,len(prior)-1,min(len(prior),max(1,count//3)),dtype=int): take(prior[i])
    # Stratify by camera height, choosing spatially spread entries in each stratum.
    bands=[[],[],[],[]]
    for r in records:
        bands[int(np.clip((r.get('camera_height_m',1.7)-0.5)/0.5,0,3))].append(r)
    remaining=count-len(chosen)
    for j,band in enumerate(bands):
        quota=remaining//4 + int(j < remaining%4)
        if band and quota:
            for i in np.linspace(0,len(band)-1,min(quota,len(band)),dtype=int): take(band[i])
    # Fill duplicate slots by farthest XY coverage; do not fall back to first-N.
    candidates=[r for r in records if r['frame_id'] not in seen]
    while candidates and len(chosen)<count:
        xy=np.array([[r.get('x',0.),r.get('y',0.)] for r in candidates])
        if chosen:
            dist=np.min([np.linalg.norm(xy-[r.get('x',0.),r.get('y',0.)],axis=1) for r in chosen],axis=0)
            idx=int(np.argmax(dist))
        else: idx=0
        take(candidates.pop(idx))
    return chosen


def find_checkpoint(name):
    p=Path(name).expanduser()
    options=[p,Path('models')/p.name,Path('weights')/p.name]
    for opt in options:
        if opt.is_file(): return opt.resolve()
    raise FileNotFoundError(f"Existing model weights not found: {name}. Pass --model /actual/path/FastSAM-s.pt. "
                            "This tool deliberately does not download or upgrade models.")


def source_code_hash():
    return digest({p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


class FastSAMPredictor:
    def __init__(self,args):
        import torch
        from ultralytics import FastSAM
        checkpoint=find_checkpoint(args.model)
        self.cfg={'imgsz':args.imgsz,'conf':args.conf,'iou':args.iou,'max_det':args.max_det,
                  'retina_masks':True,'verbose':False,'save':False,'device':str(args.device)}
        self.binding={'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                      'torch':torch.__version__,'ultralytics':importlib.metadata.version('ultralytics'),
                      'torchvision':importlib.metadata.version('torchvision'),'inference':self.cfg,
                      'mask_threshold':0.5,'float_precision':'model_default_no_half_override',
                      'package_version':__version__,'source_code_hash':source_code_hash()}
        self.torch=torch
        if str(args.device)!='cpu':
            torch.cuda.init()
            idx=int(str(args.device).replace('cuda:','').split(',')[0])
            test=torch.ones(1,device=f'cuda:{idx}'); test.sum().item(); torch.cuda.synchronize(idx)
            print('GPU:',torch.cuda.get_device_name(idx),flush=True)
        torch.set_num_threads(2)
        self.model=FastSAM(str(checkpoint))

    def __call__(self,rgb):
        # PIL is RGB; never pass RGB ndarray to an API expecting BGR.
        with self.torch.inference_mode():
            result=self.model.predict(source=Image.fromarray(rgb),**self.cfg)[0]
            if tuple(result.orig_shape)!=rgb.shape[:2]: raise RuntimeError('Model original shape mismatch')
            if result.masks is None:
                return np.empty((0,*rgb.shape[:2]),bool),np.empty(0,np.float32),np.empty((0,4),np.float32)
            masks=result.masks.data.detach().cpu().numpy()>0.5
            if masks.shape[1:]!=rgb.shape[:2]:
                raise RuntimeError(f'FastSAM returned {masks.shape}; expected (N,{rgb.shape[0]},{rgb.shape[1]}). '
                                   'retina_masks=True did not preserve alignment. STOP, do not resize blindly.')
            scores=result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            boxes=result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        if len(scores)!=len(masks) or boxes.shape!=(len(masks),4): raise RuntimeError('Mask/box count mismatch')
        if not np.isfinite(scores).all() or not np.isfinite(boxes).all(): raise RuntimeError('Model returned nonfinite proposal metadata')
        return masks,scores,boxes


def bind_run(out,plan,predictor):
    binding={'schema':'valmet.mask_archive.v1','plan_hash':plan['plan_hash'],**predictor.binding}
    binding['run_id']=digest(binding)
    if (out/'run.json').exists() and read_json(out/'run.json')!=binding:
        raise ValueError('Checkpoint, inference settings, code or library versions changed; use a NEW output directory. '
                         'Quality policy changes alone can use reselect.')
    write_json(out/'run.json',binding)
    return binding


def overlay(rgb,masks,decisions,max_draw=100):
    image=Image.fromarray(rgb).convert('RGBA')
    # Outlines preserve visible detail and overlapping/nested proposals. Never flatten IDs.
    import cv2
    canvas=np.array(image)
    colors={'keep':(65,230,120,255),'review':(255,185,40,255),'exclude':(240,70,85,255)}
    for idx,(mask,dec) in enumerate(zip(masks,decisions)):
        if idx>=max_draw: break
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas,contours,-1,colors[dec['status']],1)
    image=Image.fromarray(canvas).convert('RGB'); draw=ImageDraw.Draw(image)
    for idx,mask in enumerate(masks[:max_draw]):
        yy,xx=np.nonzero(mask)
        if len(xx):
            # Label at a real mask pixel near centroid, not outside a nonconvex mask.
            mid=np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2)
            x,y=int(xx[mid]),int(yy[mid]); text=f'{idx:03d}'
            draw.text((x,y),text,fill='white',stroke_width=1,stroke_fill='black')
    return image


def analyze_archive(dataset,out,plan,rec,p,ph,run):
    fid=rec['frame_id']; raw=out/'frames'/fid
    if not rec.get('source_ok'): return None
    if not archive_valid(raw,run['run_id'],rec['source_complete_sha256']): return None
    rgb,d,a,cam,old,_=read_source(dataset,rec,plan,verify=False)
    masks,scores,boxes=unpack_masks(raw/'masks.npz')
    if masks.shape[1:]!=d.shape: raise ValueError(f'Archive resolution mismatch: {fid}')
    finite,valid,edges=depth_support(d,a,p)
    features=image_features(rgb)
    rows=[]
    for idx,(mask,score,box) in enumerate(zip(masks,scores,boxes)):
        metrics=mask_metrics(mask,score,box,features,d,finite,valid,edges,cam,p)
        decision=decide_mask(metrics,rec,p)
        rows.append({'observation_id':f"{run['run_id']}:{fid}:m{idx:04d}",
                     'plan_hash':plan['plan_hash'],'run_id':run['run_id'],'frame_id':fid,
                     'station_id':rec['station_id'],'mask_index':idx,'mask_id':f'm{idx:04d}',
                     'raw_mask_archive':f'frames/{fid}/masks.npz','mask_shape_hw':list(d.shape),
                     'source_rgb':str(Path(dataset)/'frames'/fid/'rgb.png'),
                     'source_camera':str(Path(dataset)/'frames'/fid/'camera.json'),
                     'source_depth_m':str(Path(dataset)/'frames'/fid/'depth_m.npy'),
                     'metrics':metrics,'decision':decision,'policy_hash':ph})
    dest=out/'analysis'/ph/fid; dest.mkdir(parents=True,exist_ok=True)
    write_jsonl(dest/'observations.jsonl',rows)
    # Save the per-frame geometric support bitmap; mask boundaries are eroded separately.
    Image.fromarray((valid & ~edges).astype(np.uint8)*255).save(dest/'depth_support.png')
    with Image.open(Path(dataset)/'frames'/fid/'rgb.png') as im: thumb=im.convert('RGB'); thumb.thumbnail((480,360)); thumb.save(dest/'rgb.jpg',quality=90)
    vis=overlay(rgb,masks,[r['decision'] for r in rows]); vis.save(dest/'overlay.jpg',quality=92)
    status=Counter(r['decision']['status'] for r in rows)
    info=read_json(raw/'inference.json')
    summary={'frame_id':fid,'policy_hash':ph,'frame_status':rec['status'],'mask_count':len(rows),
             'mask_keep':status['keep'],'mask_review':status['review'],'mask_exclude':status['exclude'],
             'geometry_eligible':sum(r['decision']['eligible_geometry'] for r in rows),
             'no_masks_detected':len(rows)==0,'proposal_limit_reached':info.get('proposal_limit_reached',False),
             'source_complete_sha256':rec['source_complete_sha256'],'run_id':run['run_id']}
    write_json(dest/'summary.json',summary)
    return summary


def segment(dataset,out,plan,records,p,ph,args,predictor_factory=FastSAMPredictor):
    chosen=choose_pilot(records,args.pilot_frames) if args.pilot else records
    predictor=None; run=None; processed=0; skipped=0
    if (out/'run.json').exists(): run=read_json(out/'run.json')
    if shutil.disk_usage(out).free < 1024**3: raise RuntimeError('Keep at least 1 GiB free; actual mask archive may need several GiB')
    for idx,rec in enumerate(chosen,1):
        fid=rec['frame_id']; raw=out/'frames'/fid
        if not rec.get('source_ok') or (rec['status']=='exclude' and not args.segment_excluded):
            skipped+=1; print(f'[{idx}/{len(chosen)}] {fid}: SKIP {rec["reasons"]}',flush=True); continue
        # Construct/verify the model binding on first usable frame even when resuming.
        if predictor is None:
            predictor=predictor_factory(args); run=bind_run(out,plan,predictor)
        if archive_valid(raw,run['run_id'],rec['source_complete_sha256']):
            dest=out/'analysis'/ph/fid/'summary.json'
            if not dest.exists(): analyze_archive(dataset,out,plan,rec,p,ph,run)
            continue
        if args.limit is not None and processed>=args.limit: continue
        if shutil.disk_usage(out).free < 512*1024**2: raise RuntimeError('Low disk space; stopped safely before next image')
        start=time.perf_counter()
        rgb,d,a,cam,old,source_hash=read_source(dataset,rec,plan,verify=True)
        if source_hash!=rec['source_complete_sha256']: raise ValueError('Source changed since audit')
        masks,scores,boxes=predictor(rgb)
        raw.mkdir(parents=True,exist_ok=True); (raw/'complete.json').unlink(missing_ok=True)
        pack_masks(raw/'masks.npz',masks,scores,boxes)
        infer={'frame_id':fid,'run_id':run['run_id'],'plan_hash':plan['plan_hash'],
               'mask_count':len(masks),'shape_hw':list(rgb.shape[:2]),'created_utc':utc(),
               'proposal_limit_reached':len(masks)>=args.max_det,
               'all_returned_proposals_preserved':True,'overlaps_preserved':True,
               'source_complete_sha256':source_hash,'seconds_inference_and_archive':time.perf_counter()-start}
        write_json(raw/'inference.json',infer); mark_complete(raw,run['run_id'],source_hash)
        summary=analyze_archive(dataset,out,plan,rec,p,ph,run)
        processed+=1
        print(f'[{idx}/{len(chosen)}] {fid}: {len(masks)} masks; keep={summary["mask_keep"]}, '
              f'review={summary["mask_review"]}, exclude={summary["mask_exclude"]} | '
              f'{time.perf_counter()-start:.2f}s including QA/write',flush=True)
        del masks,scores,boxes,rgb,d,a
    print(f'New segmentation frames: {processed}; selected frames skipped by QA/integrity: {skipped}',flush=True)
    return run


def reselect(dataset,out,plan,records,p,ph):
    if not (out/'run.json').exists(): raise ValueError('No segmentation archive yet; run segment first')
    run=read_json(out/'run.json')
    for i,rec in enumerate(records,1):
        analyze_archive(dataset,out,plan,rec,p,ph,run)
        if i%100==0: print(f'Reselection [{i}/{len(records)}]',flush=True)
