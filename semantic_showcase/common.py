from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from semantic_animation.common import (read_json, write_json, sha, digest, save_npz,
    ROT, corners, read_cloud, valid_camera, camera_map_to_scene, map_to_scene)


def base_plan(path):
    path=Path(path).expanduser().resolve()
    p=read_json(path/'plan.json')
    if p.get('schema')!='valmet.semantic_animation.v1':
        raise ValueError('Expected the existing semantic_animation plan.json, not a semantic-scene directory.')
    q=dict(p);saved=q.pop('plan_id')
    if digest(q)!=saved:raise ValueError('Base plan hash changed. Use the original successful animation folder.')
    if len(p['regions'])!=p['region_count']:raise ValueError('Base region count mismatch.')
    return path,p


def checked_asset(base, region, name):
    root=Path(base).resolve();p=(root/region['asset_dir']/name).resolve()
    if not p.is_relative_to(root):raise ValueError('Asset path escapes the animation folder.')
    if not p.is_file():raise FileNotFoundError(p)
    return p


def checked_evidence(base, region, name):
    root=(Path(base)/region['asset_dir']).resolve();p=(root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file():raise FileNotFoundError(p)
    return p


def eased(x):
    """Quintic easing: stationary velocity and acceleration at endpoints."""
    x=np.clip(x,0.,1.)
    return x*x*x*(x*(x*6-15)+10)


def rz(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def scene_rotation(angle):
    return ROT.T @ rz(angle) @ ROT


def selection_ids(path, p):
    d=read_json(path)
    if d.get('base_plan_id') not in (None,p['plan_id']):raise ValueError('Selection belongs to a different base animation.')
    ids=d.get('region_ids')
    if not isinstance(ids,list) or not 1<=len(ids)<=40 or not all(isinstance(s,str) for s in ids):
        raise ValueError('Selection must contain region_ids: a list of 1–40 IDs (10–20 recommended).')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate selected region IDs are not allowed.')
    valid={r['region_id'] for r in p['regions']}
    missing=set(ids)-valid
    if missing:raise ValueError(f'Selected IDs missing from the base animation: {sorted(missing)}')
    return d,ids


def file_input_hashes(base, p):
    """Tail reads every region during its initial fade, so bind all original subsets."""
    result={}
    for r in p['regions']:
        path=checked_asset(base,r,'gaussians_original.ply')
        relative=str(path.relative_to(base));want=p['input_files_sha256'].get(relative)
        got=sha(path)
        if want and got!=want:raise ValueError('Base Gaussian asset changed: '+relative)
        result[relative]=got
    return result
