from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import colorsys
import hashlib
import json
import os
import tempfile
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_npz(path,**a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**a)
    tmp.replace(path)


def checked(path,entry):
    path=Path(path)
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
        raise ValueError(f'Integrity mismatch: {path}')


def file_records(folder,files):
    return {n:{'bytes':(Path(folder)/n).stat().st_size,'sha256':sha(Path(folder)/n)} for n in files}


def code_hash():
    return digest({p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


@contextmanager
def lock(out):
    import fcntl
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.semantic_scene.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another writer is using this scene.') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def color_for(key):
    hue=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)/0xffffffff
    return [int(255*v) for v in colorsys.hsv_to_rgb(hue,.64,.94)]


def unit(x):
    x=np.asarray(x,np.float32);n=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or np.any(n<1e-8):raise ValueError('Invalid zero/nonfinite descriptor')
    return x/n


def load_scene(out):
    out=Path(out).expanduser().resolve()
    scene=read_json(out/'scene.json')
    if scene.get('schema')!='valmet.semantic_scene.v1':raise ValueError('Not a Stage 4B scene')
    if not scene['regions']:raise ValueError('Scene has no exported regions.')
    return out,scene


def load_review(out,scene):
    path=Path(out)/'review.json'
    if not path.exists():return {'schema':'valmet.semantic_review.v1','scene_id':scene['scene_id'],'regions':{}}
    review=read_json(path)
    if review.get('scene_id')!=scene['scene_id']:raise ValueError('Review belongs to a different scene')
    return review


def load_suggestions(out,scene):
    path=Path(out)/'semantic_suggestions.json'
    if not path.exists():return {'regions':{},'note':'No automatic category suggestions generated.'}
    data=read_json(path)
    if data.get('scene_id')!=scene['scene_id']:raise ValueError('Suggestions belong to a different scene')
    return data
