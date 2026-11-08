"""Small, side-effect-free utilities for the additive animation module."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROT = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save_npz(p, **arrays):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(p)


def scene_to_map(p, scale, floor_y):
    return np.asarray(p) @ ROT.T * scale + [0,0,scale*floor_y]


def map_to_scene(p, scale, floor_y):
    return (np.asarray(p)-[0,0,scale*floor_y]) @ ROT / scale


def camera_map_to_scene(T, scale, floor_y):
    T=np.asarray(T,float);out=np.eye(4)
    out[:3,:3]=ROT.T @ T[:3,:3]
    out[:3,3]=map_to_scene(T[:3,3],scale,floor_y)
    return out


def camera_scene_to_map(T, scale, floor_y):
    T=np.asarray(T,float);out=np.eye(4)
    out[:3,:3]=ROT @ T[:3,:3]
    out[:3,3]=scene_to_map(T[:3,3],scale,floor_y)
    return out


def valid_camera(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-5):
        raise ValueError('Expected a finite 4x4 camera-to-world/map matrix.')
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=2e-3) or np.linalg.det(R)<.99:
        raise ValueError('Camera rotation is not a proper orthonormal rotation.')
    return T


def load_camera(p, scale, floor_y):
    d=read_json(p)
    if 'camera_to_map' in d: return valid_camera(d['camera_to_map'])
    if 'camera_to_world' in d: return valid_camera(camera_scene_to_map(d['camera_to_world'],scale,floor_y))
    raise ValueError('Camera JSON must contain camera_to_map or original-Gaussian camera_to_world.')


def look_at(eye, target):
    eye=np.asarray(eye,float);f=np.asarray(target,float)-eye;n=np.linalg.norm(f)
    if n<1e-8: raise ValueError('Camera eye equals target.')
    f/=n;right=np.cross(f,[0.,0.,1.])
    if np.linalg.norm(right)<1e-6: right=np.cross(f,[0.,1.,0.])
    right/=np.linalg.norm(right);up=np.cross(right,f)
    T=np.eye(4);T[:3,:3]=np.column_stack((right,-up,f));T[:3,3]=eye
    return T


def corners(lo, hi):
    return np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])


def read_cloud(p):
    with np.load(p,allow_pickle=False) as d:
        xyz=d['xyz'].astype(np.float32);rgb=d['rgb'].copy()
    if xyz.ndim!=2 or xyz.shape[1]!=3 or rgb.shape!=xyz.shape: raise ValueError(f'Invalid XYZRGB arrays: {p}')
    good=np.isfinite(xyz).all(1)&np.isfinite(rgb).all(1)
    xyz=xyz[good];rgb=rgb[good]
    if rgb.dtype.kind=='f' and rgb.size and rgb.max()<=1.001: rgb=rgb*255
    return xyz,np.rint(np.clip(rgb,0,255)).astype(np.uint8)


def sample_cloud(xyz,rgb,n,seed):
    if n>0 and len(xyz)>n:
        ix=np.sort(np.random.default_rng(seed).choice(len(xyz),n,replace=False));return xyz[ix],rgb[ix]
    return xyz,rgb


def write_ply(p,xyz,rgb):
    from plyfile import PlyData,PlyElement
    a=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,n in enumerate(('x','y','z')): a[n]=xyz[:,j]
    for j,n in enumerate(('red','green','blue')): a[n]=rgb[:,j]
    PlyData([PlyElement.describe(a,'vertex')],text=False,byte_order='<').write(str(p))


def smooth(a,b,x):
    t=np.clip((np.asarray(x)-a)/(b-a),0,1);return t*t*(3-2*t)
