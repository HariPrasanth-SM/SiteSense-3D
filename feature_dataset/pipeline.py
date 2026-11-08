from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import shutil
import time
import numpy as np
from PIL import __version__ as pillow_version
from .common import (read_json,write_json,write_jsonl,write_npz,sha,digest,source_hash,
                     is_complete,complete,json_lines)
from .preprocess import full_frame,mask_crop,patch_weights,weighted_pool,normalize_rows
from .source import audit,pilot
from . import __version__


def recipe(args):
    return dict(version=__version__, crop_size=args.crop_size,frame_side=args.frame_side,
                crop_padding_fraction=args.crop_pad,save_dense=not args.no_dense,
                mask_scope='all returned Stage 2 masks including review/exclude',
                pooling='L2 patch tokens -> area-coverage weighted mean -> L2 descriptor',
                crop_background='ImageNet mean outside raw mask',
                crop_layout='aspect-preserving letterbox; no center crop; padded bbox',
                normalization={'mean':[.485,.456,.406],'std':[.229,.224,.225]},
                output_storage='float16; promote and renormalize for similarity',
                numpy=np.__version__,pillow=pillow_version,
                code_sha256=source_hash(Path(__file__).parent))


def infer_frame(encoder,rgb,masks,args):
    n,h,w=masks.shape;d=encoder.dim
    frame_tensor,frame_spec=full_frame(rgb,args.frame_side)
    patch_batch,global_cls=encoder.infer(frame_tensor[None])
    patches=patch_batch[0]
    weights=np.stack([patch_weights(m,frame_spec) for m in masks]) if n else np.empty((0,len(patches)),np.float32)
    frame_pool,frame_valid,frame_mass,frame_eff=weighted_pool(patches,weights)
    crop_pool=np.zeros((n,d),np.float32);crop_cls=np.zeros((n,d),np.float32)
    crop_valid=np.zeros(n,bool);cls_valid=np.zeros(n,bool)
    crop_mass=np.zeros(n,np.float32);crop_eff=np.zeros(n,np.float32)
    crop_specs=[None]*n
    for start in range(0,n,args.batch_size):
        tensors=[];wp=[];ids=[]
        for i in range(start,min(start+args.batch_size,n)):
            res=mask_crop(rgb,masks[i],args.crop_size,args.crop_pad)
            if res is None: continue
            t,m,spec=res;tensors.append(t);wp.append(m);ids.append(i);crop_specs[i]=spec
        if not ids: continue
        p,c=encoder.infer(np.stack(tensors))
        cl,clok=normalize_rows(c)
        for b,i in enumerate(ids):
            v,ok,mass,eff=weighted_pool(p[b],wp[b])
            crop_pool[i]=v[0];crop_valid[i]=ok[0];crop_mass[i]=mass[0];crop_eff[i]=eff[0]
            crop_cls[i]=cl[b];cls_valid[i]=clok[b]
    arrays={'mask_indices':np.arange(n,dtype=np.int32),
            'crop_mask_pool':crop_pool.astype(np.float16),'crop_cls':crop_cls.astype(np.float16),
            'frame_mask_pool':frame_pool.astype(np.float16),
            'crop_valid':crop_valid,'cls_valid':cls_valid,'frame_valid':frame_valid,
            'crop_patch_mass':crop_mass,'crop_effective_patches':crop_eff,
            'frame_patch_mass':frame_mass,'frame_effective_patches':frame_eff}
    pn,_=normalize_rows(patches);gc,_=normalize_rows(global_cls)
    ch,cw=frame_spec['canvas_hw']
    dense={'patch_tokens':pn.reshape(ch//16,cw//16,d).astype(np.float16),
           'cls':gc[0].astype(np.float16),'patch_size':np.array(16,dtype=np.int32)}
    return arrays,dense,{'frame_transform':frame_spec,'crop_transforms':crop_specs,
        'empty_masks':[int(i) for i in np.flatnonzero(~masks.any(axis=(1,2)))],
        'crop_small_patch_support':[int(i) for i in np.flatnonzero(crop_eff<2)],
        'frame_small_patch_support':[int(i) for i in np.flatnonzero(frame_eff<2)],
        'notes':['No semantic labels or instance matches assigned.',
                 'Contextual patch tokens can include neighboring/background information.',
                 'A pooled descriptor does not prove complete object segmentation.']}


def bind(out,source,encoder,args):
    config={'schema':'valmet.dinov3_features.v1','source':source.binding,
            'model':encoder.identity,'recipe':recipe(args)}
    config['run_id']=digest(config)
    path=out/'run.json'
    if path.exists() and read_json(path)!=config:
        raise ValueError('Model, precision, preprocessing, source or feature code changed. Use a NEW feature output folder.')
    write_json(path,config)
    return config


def extract(source,out,args,encoder_factory=None):
    if encoder_factory is None:
        from .model import DINOEncoder
        encoder_factory=DINOEncoder
    records=audit(source,out)
    selected=pilot(records,args.pilot_frames) if args.pilot else [r for r in records if r['segmented']]
    encoder=encoder_factory(args.repo,args.weights,args.arch,args.device,args.precision)
    run=bind(out,source,encoder,args)
    write_json(out/'pilot_selection.json', {'frames':[r['frame_id'] for r in selected]} if args.pilot else
               {'note':'Full run; pilot subset can be recreated from input inventory.'})
    done=0;reused=0;since=time.perf_counter()
    for j,rec in enumerate(selected,1):
        fid=rec['frame_id'];folder=out/'frames'/fid
        if is_complete(folder,run['run_id'],rec['input_hash']): reused+=1;continue
        if args.limit is not None and done>=args.limit: break
        if shutil.disk_usage(out).free<512*1024**2: raise RuntimeError('Less than 512 MiB free; stopped before next frame.')
        t=time.perf_counter();rgb,masks=source.load_pixels(rec)
        folder.mkdir(parents=True,exist_ok=True);(folder/'complete.json').unlink(missing_ok=True)
        # Source remains immutable. A failed frame is recomputed on resume; no partial arrays are indexed.
        arrays,dense,meta=infer_frame(encoder,rgb,masks,args)
        write_npz(folder/'features.npz',**arrays)
        if not args.no_dense: write_npz(folder/'dense.npz',**dense)
        meta.update(frame_id=fid,run_id=run['run_id'],input_hash=rec['input_hash'],mask_count=len(masks),
                    original_hw=list(rgb.shape[:2]),feature_dimension=encoder.dim,
                    source_camera=str(source.dataset/'frames'/fid/'camera.json'),
                    seconds_compute_and_write=time.perf_counter()-t)
        write_json(folder/'metadata.json',meta);complete(folder,run['run_id'],rec['input_hash'])
        done+=1
        print(f'[{j}/{len(selected)}] {fid}: archived {len(masks)} mask descriptors | '
              f'{int(arrays["crop_valid"].sum())} valid crops | {time.perf_counter()-t:.2f}s',flush=True)
        del masks,rgb,arrays,dense
    print(f'New frames {done}, reused {reused}; {time.perf_counter()-since:.1f}s after model load.',flush=True)
    return build_index(source,out,records)


def eligible(row,frame_status,valid):
    # Default deliberately stricter than Stage 2: no review FRAME or review MASK seeds.
    accepted=(frame_status=='keep' and row['decision']['status']=='keep'
              and bool(row['decision']['eligible_features']) and bool(valid))
    return bool(accepted), bool(accepted and row['decision']['eligible_geometry'])


def build_index(source,out,records=None):
    if records is None: records=[source.inspect(v) for v in source.frames]
    run=read_json(out/'run.json');counts=Counter();byte_count=0
    names=('observations.jsonl','association_candidates.jsonl','geometry_candidates.jsonl','quarantined_observations.jsonl')
    handles={n:(out/(n+'.tmp')).open('w') for n in names}
    try:
        for rec in records:
            counts['planned_frames']+=1
            if not rec['segmented']:
                counts['frames_without_masks']+=1;continue
            counts['segmented_frames']+=1;counts['source_masks']+=rec['mask_count']
            fid=rec['frame_id'];folder=out/'frames'/fid
            if not is_complete(folder,run['run_id'],rec['input_hash']):
                counts['remaining_frames']+=1;counts['remaining_masks']+=rec['mask_count'];continue
            rows=source.read_rows(fid)
            with np.load(folder/'features.npz',allow_pickle=False) as a:
                if not np.array_equal(a['mask_indices'],np.arange(len(rows))): raise ValueError('Feature row IDs mismatch')
                feature_valid=a['crop_valid'].copy();frame_valid=a['frame_valid'].copy()
                support=a['crop_effective_patches'].copy();full_support=a['frame_effective_patches'].copy()
                if a['crop_mask_pool'].shape!=(len(rows),run['model']['dimension']): raise ValueError('Unexpected descriptor shape')
            counts['extracted_frames']+=1
            byte_count+=sum((folder/n).stat().st_size for n in ('features.npz','dense.npz') if (folder/n).exists())
            for i,row in enumerate(rows):
                accept,geom=eligible(row,rec['frame_status'],feature_valid[i])
                reason=[]
                if rec['frame_status']!='keep': reason.append('frame_not_keep')
                if row['decision']['status']!='keep': reason.append('mask_not_keep')
                if not feature_valid[i]: reason.append('empty_or_invalid_descriptor')
                if not row['decision']['eligible_features']: reason.append('stage2_feature_ineligible')
                r={'observation_id':row['observation_id'],'feature_run_id':run['run_id'],
                   'frame_id':fid,'station_id':rec['station_id'],'mask_index':i,'feature_row':i,
                   'feature_file':f'frames/{fid}/features.npz','default_descriptor':'crop_mask_pool',
                   'feature_valid':bool(feature_valid[i]),'frame_descriptor_valid':bool(frame_valid[i]),
                   'mask_status':row['decision']['status'],'frame_status':rec['frame_status'],
                   'eligible_association':accept,'eligible_geometry':geom,'quarantine_reasons':reason,
                   'quality_policy_hash':source.policy,'stage2_analysis_file':str(source.seg/'analysis'/source.policy/fid/'observations.jsonl'),
                   'source_mask_archive':str(source.seg/'frames'/fid/'masks.npz'),
                   'source_rgb':str(source.dataset/'frames'/fid/'rgb.png'),
                   'source_camera':str(source.dataset/'frames'/fid/'camera.json'),
                   'source_depth_m':str(source.dataset/'frames'/fid/'depth_m.npy'),
                   'partial_surface_summary':row['metrics'].get('geometry'),
                   'quality_hints':row['decision'].get('hints',[]),
                   'crop_effective_patches':float(support[i]),'frame_effective_patches':float(full_support[i]),
                   'semantic_label':None,'object_id':None}
                line=json.dumps(r,separators=(',',':'),allow_nan=False)+'\n'
                handles['observations.jsonl'].write(line)
                handles['association_candidates.jsonl' if accept else 'quarantined_observations.jsonl'].write(line)
                if geom: handles['geometry_candidates.jsonl'].write(line)
                counts['archived_masks']+=1;counts['source_mask_'+r['mask_status']]+=1
                counts['valid_crop_features']+=bool(feature_valid[i]);counts['association_candidates']+=accept
                counts['geometry_candidates']+=geom;counts['quarantined']+=not accept
    finally:
        for f in handles.values():f.close()
    for n in names:(out/(n+'.tmp')).replace(out/n)
    summary={'schema':'valmet.stage3.summary.v1','run_id':run['run_id'],'counts':dict(counts),
             'feature_archive_bytes':byte_count,'quality_policy_hash':source.policy,
             'selection':'mask keep AND frame keep AND valid crop descriptor; geometry also requires Stage 2 geometry eligibility',
             'notes':['Descriptors archived for all existing masks; no mask deleted.',
                      'Overlapping masks stay independent. Labels and object IDs remain null.',
                      'No cross-view merge, object point-cloud extraction, navigation or text lookup yet.',
                      'Cosine similarity is appearance similarity, not probability of same physical instance.']}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary['counts'],indent=2),flush=True)
    return summary
