from __future__ import annotations
import hashlib, io, json, math, os, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image

def read_json(path):
    return json.loads(Path(path).read_text())

def jsonable(x):
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    return x

def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f: json.dump(jsonable(data),f,indent=2,allow_nan=False)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def image_bytes(rgb, fmt='JPEG', **kw):
    f=io.BytesIO(); Image.fromarray(np.asarray(rgb,dtype=np.uint8)).save(f,format=fmt,**kw);return f.getvalue()

def number(value, lo, hi, name):
    v=float(value)
    if not math.isfinite(v) or not lo<=v<=hi: raise ValueError(f'{name} must be between {lo} and {hi}.')
    return v
