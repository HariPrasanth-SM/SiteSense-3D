from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .quality import validate_camera

VERSION='valmet.mask_archive.v1'
SOURCE_FILES=('rgb.png','depth_scene.npy','depth_m.npy','alpha.npy','valid.png','camera.json','qa.json')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def write_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row,allow_nan=False)+'\n')
    os.replace(tmp,path)

def load_plan(dataset):
    p=read_json(Path(dataset)/'plan.json')
    if p.get('schema')!='valmet.calibrated_views.v1': raise ValueError('Unsupported capture schema')
    if p.get('plan_hash')!=digest({k:v for k,v in p.items() if k!='plan_hash'}): raise ValueError('Capture plan hash mismatch')
    ids=[v['frame_id'] for v in p['frames']]
    if len(set(ids))!=len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',v) for v in ids):
        raise ValueError('Invalid or duplicated frame IDs')
    return p

def verify_source(folder,plan_hash):
    stamp=read_json(folder/'complete.json')
    if stamp['plan_hash']!=plan_hash or stamp['frame_id']!=folder.name: raise ValueError('Capture completion marker mismatch')
    for n in SOURCE_FILES:
        info=stamp['files'][n]; path=folder/n
        if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
            raise ValueError(f'Source checksum mismatch: {path}')
    return sha(folder/'complete.json')

def read_source(dataset,view,plan,verify=True):
    folder=Path(dataset)/'frames'/view['frame_id']
    stamp_hash=verify_source(folder,plan['plan_hash']) if verify else sha(folder/'complete.json')
    with Image.open(folder/'rgb.png') as im: rgb=np.asarray(im.convert('RGB')).copy()
    depth=np.load(folder/'depth_m.npy',allow_pickle=False)
    alpha=np.load(folder/'alpha.npy',allow_pickle=False)
    camera=read_json(folder/'camera.json')
    if depth.ndim!=2 or rgb.shape!=(*depth.shape,3) or alpha.shape!=depth.shape: raise ValueError('Source array shape mismatch')
    if not np.issubdtype(depth.dtype,np.floating) or not np.issubdtype(alpha.dtype,np.floating): raise ValueError('Depth/alpha must be floating arrays')
    validate_camera(camera,depth.shape,view['frame_id'],plan['plan_hash'])
    # Prevent a source with wrong scale from silently influencing anomaly thresholds.
    if not np.isclose(camera['depth_scale_scene_to_metres'],plan['config']['scale']): raise ValueError('Depth-scale mismatch')
    return rgb,depth,alpha,camera,read_json(folder/'qa.json'),stamp_hash

def pack_masks(path,masks,scores,boxes):
    masks=np.asarray(masks,dtype=bool)
    if masks.ndim!=3: raise ValueError('Masks must be (N,H,W)')
    n,h,w=masks.shape
    if np.shape(scores)!=(n,) or np.shape(boxes)!=(n,4): raise ValueError('Proposal metadata count mismatch')
    bits=np.packbits(masks.reshape(n,h*w),axis=1,bitorder='little')
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,bits=bits,shape=np.array([n,h,w],dtype=np.int64),
                            scores=np.asarray(scores,np.float32),boxes=np.asarray(boxes,np.float32),bitorder=np.array('little'))
    os.replace(tmp,path)

def unpack_masks(path):
    with np.load(path,allow_pickle=False) as d:
        n,h,w=map(int,d['shape']); bits=d['bits']; scores=d['scores'].copy(); boxes=d['boxes'].copy()
        if str(d['bitorder'])!='little' or bits.shape!=(n,(h*w+7)//8): raise ValueError('Bad mask archive')
        masks=np.unpackbits(bits,axis=1,count=h*w,bitorder='little').reshape(n,h,w).astype(bool)
    return masks,scores,boxes

def archive_valid(folder,run_id,source_hash,verify=True):
    try:
        meta=read_json(folder/'complete.json')
        if meta['run_id']!=run_id or meta['source_complete_sha256']!=source_hash: return False
        for name,rec in meta['files'].items():
            p=folder/name
            if p.stat().st_size!=rec['bytes'] or (verify and sha(p)!=rec['sha256']): return False
        return True
    except (OSError,ValueError,KeyError,TypeError): return False

def mark_complete(folder,run_id,source_hash):
    files={n:{'bytes':(folder/n).stat().st_size,'sha256':sha(folder/n)} for n in ('masks.npz','inference.json')}
    write_json(folder/'complete.json',{'run_id':run_id,'source_complete_sha256':source_hash,'files':files})

@contextmanager
def output_lock(out):
    import fcntl
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with (out/'.stage2.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process is writing this Stage 2 output; use one process')
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
