#!/usr/bin/env python3
"""Read-only asset/binding/frame inspector for the existing Valmet project.

Writes ONLY to a new inspection output directory. No scene, map, binding, or web
code edits. Point clouds are already in map coordinates; original Gaussian PLYs
remain in their source coordinate frame.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

R_GM = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
MODES = ('binding', 'region_center', 'region_surface', 'workbook_raw',
         'workbook_as_scene', 'binding_as_scene', 'undo_double_transform', 'swap_xy')
COLORS = {'binding': '#ea5455', 'region_center': '#d7a020', 'region_surface': '#00886f',
          'workbook_raw': '#9873aa', 'workbook_as_scene': '#d97419',
          'binding_as_scene': '#2568ca', 'undo_double_transform': '#219a9a', 'swap_xy': '#a25c76'}


def load_json(path):
    return json.loads(Path(path).read_text())


def jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    return x


def save_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def vec3(x):
    try:
        a = np.asarray(x, dtype=float)
        return a if a.shape == (3,) and np.isfinite(a).all() else None
    except (TypeError, ValueError): return None


def scene_to_map(points, scale, floor):
    return np.asarray(points, float) @ R_GM.T * scale + [0., 0., scale * floor]


def map_to_scene(points, scale, floor):
    return (np.asarray(points, float) - [0., 0., scale * floor]) @ R_GM / scale


def camera_to_map(c2g, scale, floor):
    c2m = np.eye(4)
    c2m[:3, :3] = R_GM @ c2g[:3, :3]
    c2m[:3, 3] = scene_to_map(c2g[:3, 3], scale, floor)
    return c2m


def bounds(xyz):
    if not len(xyz): return None
    lo, hi = xyz.min(0), xyz.max(0)
    return dict(min=lo, max=hi, center=(lo + hi) / 2, extent=hi - lo)


def rgb_uint8(rgb):
    rgb = np.asarray(rgb)
    if np.issubdtype(rgb.dtype, np.floating) and rgb.size and np.nanmax(rgb) <= 1.0001:
        rgb = rgb * 255.
    return np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)


def load_cloud(path):
    with np.load(path, allow_pickle=False) as f:
        xyz = np.asarray(f['xyz'], dtype=np.float64)
        rgb = rgb_uint8(f['rgb']) if 'rgb' in f else np.full(xyz.shape, 180, np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError(f'Invalid XYZRGB arrays: {path}')
    ok = np.isfinite(xyz).all(1)
    return xyz[ok], rgb[ok]


def write_ply(path, xyz, rgb):
    """Binary XYZRGB surface/center cloud; never masquerades as a Gaussian PLY."""
    xyz = np.asarray(xyz, np.float32); rgb = rgb_uint8(rgb)
    data = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, name in enumerate(('x', 'y', 'z')): data[name] = xyz[:, k]
    for k, name in enumerate(('red', 'green', 'blue')): data[name] = rgb[:, k]
    header = ('ply\nformat binary_little_endian 1.0\ncomment map-frame diagnostic XYZRGB\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as f: f.write(header.encode()); f.write(data.tobytes())


def save_cloud(folder, stem, xyz, rgb):
    np.savez_compressed(folder / (stem + '.npz'), xyz=np.asarray(xyz, np.float32), rgb=rgb_uint8(rgb))
    write_ply(folder / (stem + '.ply'), xyz, rgb)


class MapData:
    def __init__(self, path):
        self.path = Path(path).resolve(); self.cfg = yaml.safe_load(self.path.read_text())
        if self.cfg.get('mode', 'trinary') != 'trinary': raise ValueError('Only trinary maps supported')
        image = Path(self.cfg['image']).expanduser()
        self.image_path = image if image.is_absolute() else self.path.parent / image
        a = np.asarray(Image.open(self.image_path).convert('RGB'), float).mean(2) / 255.
        p = a if self.cfg.get('negate', 0) else 1. - a
        g = np.full(a.shape, -1, np.int8)
        g[p < float(self.cfg['free_thresh'])] = 0
        g[p > float(self.cfg['occupied_thresh'])] = 100
        self.grid = np.flipud(g).copy(); self.h, self.w = g.shape
        self.res = float(self.cfg['resolution']); self.origin = np.asarray(self.cfg['origin'], float)
        if self.res <= 0 or self.origin.shape != (3,): raise ValueError('Invalid map metadata')
        co, si = math.cos(self.origin[2]), math.sin(self.origin[2])
        self.rot = np.array([[co, -si], [si, co]])

    def grid_xy(self, xy):
        return (np.asarray(xy, float) - self.origin[:2]) @ self.rot / self.res

    def world(self, cr):
        return np.asarray(cr, float) * self.res @ self.rot.T + self.origin[:2]

    def classify(self, p):
        if p is None: return {'state': 'unavailable'}
        col, row = np.floor(self.grid_xy(p[:2])).astype(int)
        inside = 0 <= row < self.h and 0 <= col < self.w
        return {'row': int(row), 'col': int(col), 'inside_map': bool(inside),
                'state': {-1: 'unknown', 0: 'free', 100: 'occupied'}[int(self.grid[row, col])] if inside else 'outside'}

    def image(self):
        a = np.full((self.h, self.w, 3), [117, 128, 141], np.uint8)
        a[self.grid == 0] = [204, 222, 185]; a[self.grid == 100] = [36, 43, 52]
        return Image.fromarray(np.flipud(a))


class Sources:
    def __init__(self, args):
        self.bundle = Path(args.bundle).expanduser().resolve()
        self.config = load_json(self.bundle / 'config.json')
        self.db = load_json(self.bundle / 'workbook.json')
        self.bindings = load_json(self.bundle / 'asset_bindings.json')
        self.region_root = Path(args.regions or self.config['regions']).expanduser().resolve()
        en = args.enhancement or self.config.get('enhancement')
        self.enh_root = Path(en).expanduser().resolve() if en else None
        self.map = MapData(args.map or self.config['map'])
        self.scene = load_json(self.region_root / 'scene.json')
        self.enh = load_json(self.enh_root / 'config.json') if self.enh_root and (self.enh_root / 'config.json').exists() else {}
        self.scale = float(self.config['scale']); self.floor = float(self.config['floor_y'])
        self.review = load_json(self.region_root / 'review.json').get('regions', {}) if (self.region_root / 'review.json').exists() else {}
        self.env_path = Path(args.environment).resolve() if args.environment else None
        self.env = None
        self.monitor = {}
        for p in [self.bundle / 'config.json', self.bundle / 'workbook.json', self.bundle / 'asset_bindings.json',
                  self.map.path, self.map.image_path, self.region_root / 'scene.json']:
            self.monitor[str(p)] = digest(p)

    def unchanged(self):
        return all(digest(p) == h for p, h in self.monitor.items())

    def region(self, rid):
        if not rid: return None, None, None
        folder = self.region_root / 'regions' / rid
        if not (folder / 'region.json').exists(): return None, folder, None
        r = load_json(folder / 'region.json')
        en = self.enh_root / 'regions' / rid if self.enh_root else None
        return r, folder, en

    def cloud(self, rid):
        r, raw, en = self.region(rid)
        if not r: raise ValueError(f'Region not found: {rid}')
        path = en / 'surface_clean.npz' if en else Path('__missing__')
        if not path.is_file(): path = raw / 'surface.npz'
        return (*load_cloud(path), path)

    def environment(self):
        if self.env is None and self.env_path is not None:
            self.env = load_cloud(self.env_path)
        return self.env


def coordinate_candidates(binding, anchor, box=None, xyz=None):
    p = vec3(binding.get('map_xyz'))
    a = None
    if isinstance(anchor, dict):
        a = vec3([anchor.get('x_local_m'), anchor.get('y_local_m'), anchor.get('z_local_m')])
    options = {}
    if p is not None: options['binding'] = p
    if box: options['region_center'] = np.asarray(box['center'], float)
    if xyz is not None and len(xyz):
        med = np.median(xyz, axis=0)
        options['region_surface'] = xyz[np.argmin(np.sum((xyz - med) ** 2, axis=1))]
    if a is not None: options['workbook_raw'] = a
    return options


def all_candidates(options, scale, floor):
    o = {k: np.asarray(v, float) for k, v in options.items()}
    if 'binding' in o:
        o['binding_as_scene'] = scene_to_map(o['binding'], scale, floor)
        o['undo_double_transform'] = map_to_scene(o['binding'], scale, floor)
        o['swap_xy'] = o['binding'][[1, 0, 2]].copy()
    if 'workbook_raw' in o: o['workbook_as_scene'] = scene_to_map(o['workbook_raw'], scale, floor)
    return o


DESCRIPTIONS = {
    'binding': 'Current saved dashboard target, already map XYZ. Baseline; no extra transform.',
    'region_center': 'Recomputed center of current cleaned/raw region bounds. Diagnostic for stale stored target.',
    'region_surface': 'Actual retained surface point nearest the region median. Diagnostic for center lying in empty space.',
    'workbook_raw': 'Original synthetic anchor interpreted as map XYZ ONLY AS A HYPOTHESIS; NOT registered to scan.',
    'workbook_as_scene': 'Original synthetic anchor interpreted as Gaussian XYZ then transformed. NOT justified registration.',
    'binding_as_scene': 'Apply scene-to-map to binding: only relevant if saved map_xyz was actually raw Gaussian XYZ.',
    'undo_double_transform': 'Apply inverse transform to binding: only relevant if scene-to-map was accidentally applied twice.',
    'swap_xy': 'Swap horizontal axes of the binding. Diagnostic reflection, NOT an approved rigid scene transform.'}


def source_global_checks(s):
    checks = []
    def check(name, passed, detail): checks.append({'check': name, 'pass': bool(passed), 'detail': detail})
    check('rotation_det_plus_one', np.isclose(np.linalg.det(R_GM), 1), float(np.linalg.det(R_GM)))
    scale2 = s.scene.get('scale_m_per_scene_unit'); floor2 = s.scene.get('floor_y_gaussian')
    if scale2 is not None: check('dashboard_scale_matches_region_scene', np.isclose(s.scale, scale2), [s.scale, scale2])
    if floor2 is not None: check('dashboard_floor_matches_region_scene', np.isclose(s.floor, floor2), [s.floor, floor2])
    for key, value in [('scale', s.scale), ('floor_y', s.floor)]:
        if key in s.enh: check('dashboard_' + key + '_matches_enhancement', np.isclose(value, s.enh[key]), [value, s.enh[key]])
    if s.config.get('map_sha256'): check('map_yaml_matches_prepare_hash', digest(s.map.path) == s.config['map_sha256'], str(s.map.path))
    a = s.map.world([.5, .5]); b = s.map.grid_xy(a)
    check('map_first_cell_roundtrip', np.allclose(b, [.5, .5]), b.tolist())
    return checks


def audit_record(s, aid, with_cloud=False):
    asset = s.db['assets'][aid]; b = s.bindings['assets'].get(aid, {})
    r, folder, en = s.region(b.get('region_id')); warnings = []
    meta = load_json(en / 'enhancement.json') if en and (en / 'enhancement.json').exists() else {}
    xyz = rgb = None; cloud_path = None
    box = meta.get('clean_bbox', r.get('bbox')) if r else None
    stored_box = box
    if with_cloud and r:
        xyz, rgb, cloud_path = s.cloud(b['region_id']); box = bounds(xyz)
    choices = all_candidates(coordinate_candidates(b, asset.get('anchor_original'), box, xyz), s.scale, s.floor)
    current = choices.get('binding'); rid = b.get('region_id')
    if not rid: warnings.append('NO_BOUND_REGION: map pin/anchor cannot extract semantic object without an association.')
    if rid and r is None: warnings.append('MISSING_REGION: region ID cannot be resolved in current scene.')
    if b.get('status') == 'procedural_demo_binding': warnings.append('PROCEDURAL_IDENTITY: label/class was not matched to visual equipment.')
    if asset.get('anchor_original', {}).get('mapping_status') == 'placeholder_until_valmet_scene_received':
        warnings.append('WORKBOOK_PLACEHOLDER: original anchor is not a registered scan coordinate.')
    delta = None
    if current is not None and box:
        delta = float(np.linalg.norm(current - box['center']))
        if delta > .10: warnings.append('TARGET_DIFFERS_FROM_CURRENT_REGION_CENTER: inspect stale binding or intentional manual aim point.')
    if not rid and current is not None and np.isclose(current[2], 1.):
        warnings.append('FIXED_PIN_HEIGHT: supplied web map-click path sets Z=1.0; verify actual target height.')
    if stored_box and box and with_cloud:
        meta_delta = float(np.linalg.norm(np.asarray(stored_box['center']) - box['center']))
        if meta_delta > .02: warnings.append('BBOX_METADATA_DIFFERS_FROM_POINT_DATA.')
    else: meta_delta = None
    choice_records = {}
    for name, p in choices.items():
        d = {'map_xyz_hypothesis': p, 'map': s.map.classify(p), 'description': DESCRIPTIONS[name]}
        if xyz is not None and len(xyz): d['nearest_retained_surface_distance_m'] = float(np.min(np.linalg.norm(xyz-p,axis=1)))
        if box: d['distance_to_current_bbox_center_m'] = float(np.linalg.norm(p-box['center']))
        choice_records[name] = d
    if current is not None and not s.map.classify(current).get('inside_map', False): warnings.append('CURRENT_TARGET_OUTSIDE_MAP.')
    rec = {'asset_id': aid, 'tag': asset['master'].get('tag'), 'asset_class': asset['master'].get('asset_class'),
           'binding': b, 'anchor_original': asset.get('anchor_original'), 'region_id': rid,
           'reviewed_region_label': s.review.get(rid, {}).get('semantic_label'), 'region_status': r.get('status') if r else None,
           'current_region_bbox': box, 'binding_center_difference_m': delta, 'bbox_metadata_difference_m': meta_delta,
           'coordinate_hypotheses': choice_records, 'warnings': warnings,
           'cloud_source': str(cloud_path) if cloud_path else None, 'source_views': r.get('views', []) if r else []}
    return rec, xyz, rgb


def font(size=16):
    # Use an installed font if present. No font files are shipped.
    try: return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError: return ImageFont.load_default()


def map_plot(s, records, out, selected=None, mode='binding'):
    scale = min(5., 1000/max(s.map.h, s.map.w)); w = int(s.map.w*scale); h = int(s.map.h*scale)
    canvas = Image.new('RGB', (w+40,h+90), 'white'); canvas.paste(s.map.image().resize((w,h),Image.Resampling.NEAREST),(20,50))
    d=ImageDraw.Draw(canvas); d.text((20,10),f'Asset coordinates: {mode} (map frame)',font=font(18),fill='black')
    outside=[]
    for rec in records:
        option=rec['coordinate_hypotheses'].get(mode)
        if not option: continue
        c,r=s.map.grid_xy(option['map_xyz_hypothesis'][:2]); x=20+c*scale; y=50+(s.map.h-r)*scale
        if not (0<=c<s.map.w and 0<=r<s.map.h): outside.append(rec['asset_id']); continue
        col=COLORS.get(mode,'#ea5455'); radius=5 if selected is None or rec['asset_id']==selected else 3
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=col,outline='white',width=1)
        d.text((x+6,y-12),rec['tag'] or rec['asset_id'],font=font(10),fill='white',stroke_width=1,stroke_fill='black')
    d.text((20,h+58),f'Outside map: {len(outside)} | occupied target does not imply transform error',font=font(12),fill='black')
    canvas.save(out)


def export_overview(s, records, out, checks):
    save_json(out/'audit.json',{'schema':'sitesense.asset_inspection.v1','config':s.config,'global_checks':checks,
        'coordinate_equation':'p_map = scale * [[1,0,0],[0,0,1],[0,-1,0]] * p_gaussian + [0,0,scale*floor_y]',
        'asset_count':len(records),'assets':records,
        'limits':['Read-only diagnosis. Does not identify physical equipment from metadata.',
                  'Near geometry does not verify asset identity. Hypotheses are not applied to source data.',
                  'No guarantee of collision clearance or target visibility. Scale remains provisional.']})
    columns=['asset_id','tag','asset_class','binding_status','region_id','target_x','target_y','target_z',
             'map_cell','binding_center_difference_m','reviewed_region_label','warnings']
    with (out/'assets.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in records:
            p=vec3(r['binding'].get('map_xyz')); pos=p if p is not None else [None]*3
            w.writerow(dict(asset_id=r['asset_id'],tag=r['tag'],asset_class=r['asset_class'],
                binding_status=r['binding'].get('status'),region_id=r['region_id'],target_x=pos[0],target_y=pos[1],target_z=pos[2],
                map_cell=r['coordinate_hypotheses'].get('binding',{}).get('map',{}).get('state'),
                binding_center_difference_m=r['binding_center_difference_m'],reviewed_region_label=r['reviewed_region_label'],
                warnings=' | '.join(r['warnings'])))
    map_plot(s,records,out/'assets_current_map.png')
    map_plot(s,records,out/'workbook_placeholder_map.png',mode='workbook_raw')


def independent_view_check(s, rec, xyz, v, out, index, max_points=3000):
    """Measure projection/depth agreement; avoid presenting self-projection as external validation."""
    folder=Path(v['source_frame']); cam=load_json(folder/'camera.json')
    rgb=np.asarray(Image.open(folder/'rgb.png').convert('RGB')); h,w=rgb.shape[:2]
    with np.load(v['mask_archive'],allow_pickle=False) as z:
        n,mh,mw=map(int,z['shape']); order=str(z['bitorder']); idx=int(v['mask_index'])
        if order!='little' or not 0<=idx<n: raise ValueError('Invalid mask archive')
        mask=np.unpackbits(z['bits'][idx],count=mh*mw,bitorder='little').reshape(mh,mw).astype(bool)
    if mask.shape!=(h,w): raise ValueError('RGB and mask resolutions differ; no resizing attempted')
    K=np.asarray(cam['K'],float); c2m=np.asarray(cam.get('camera_to_map',v.get('camera_to_map')),float)
    if c2m.shape!=(4,4): raise ValueError('Missing map-frame camera matrix')
    measurements={'frame_id':v['frame_id'],'mask_index':idx,'station_id':v['station_id'],
        'rotation_det':float(np.linalg.det(c2m[:3,:3])),
        'rotation_orthogonality_error':float(np.linalg.norm(c2m[:3,:3].T@c2m[:3,:3]-np.eye(3))),
        'note':'Overlap is diagnostic on original supporting views; not an independent accuracy measurement.'}
    if 'camera_to_world' in cam:
        c2g=np.asarray(cam['camera_to_world'],float); predicted=camera_to_map(c2g,s.scale,s.floor)
        measurements['camera_translation_discrepancy_m']=float(np.linalg.norm(predicted[:3,3]-c2m[:3,3]))
        measurements['camera_rotation_discrepancy']=float(np.linalg.norm(predicted[:3,:3]-c2m[:3,:3]))
    if 'scene_to_map' in cam:
        sm=cam['scene_to_map'];measurements['saved_scene_to_map']=sm
        measurements['scale_matches_dashboard']=bool(np.isclose(float(sm['scale']),s.scale))
        f=sm.get('floor_y_scene');measurements['floor_matches_dashboard']=None if f is None else bool(np.isclose(f,s.floor))
    if 'world_to_camera' in cam and 'camera_to_world' in cam:
        measurements['world_camera_inverse_error']=float(np.max(np.abs(np.asarray(cam['world_to_camera'])@np.asarray(cam['camera_to_world'])-np.eye(4))))
    frame_depth=np.load(folder/'depth_m.npy',allow_pickle=False)
    alpha=np.load(folder/'alpha.npy',allow_pickle=False)
    if frame_depth.shape!=(h,w) or alpha.shape!=(h,w): raise ValueError('Depth/RGB shape mismatch')
    sample=xyz[np.linspace(0,len(xyz)-1,min(max_points,len(xyz))).astype(int)]
    pc=(sample-c2m[:3,3])@c2m[:3,:3]; q=pc@K.T
    good=np.isfinite(pc).all(1)&(pc[:,2]>1e-6)
    uv=np.full((len(pc),2),-1.);uv[good]=q[good,:2]/q[good,2:3]
    ij=np.floor(np.clip(uv,-1e6,1e6)).astype(int)
    visible=good&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    ids=np.flatnonzero(visible); hit=np.zeros(len(pc),bool); agree=np.zeros(len(pc),bool)
    if len(ids):
        uu,vv=ij[ids,0],ij[ids,1]; dd=frame_depth[vv,uu]
        hit[ids]=mask[vv,uu]
        agree[ids]=np.isfinite(dd)&(dd>0)&(alpha[vv,uu]>=.5)&(np.abs(pc[ids,2]-dd)<(.12+.03*dd))
    measurements.update(surface_sample_count=len(pc),positive_depth_fraction=float(good.mean()),
        inside_image_fraction=float(visible.mean()),
        mask_fraction_among_in_image=float(hit[ids].mean()) if len(ids) else None,
        depth_agreement_among_in_image=float(agree[ids].mean()) if len(ids) else None,
        hypotheses={})
    im=rgb.copy();im[mask]=(.7*im[mask]+.3*np.array([0,255,100])).astype(np.uint8)
    image=Image.fromarray(im);d=ImageDraw.Draw(image)
    for j in ids[::max(1,len(ids)//1000)]:
        u,vp=uv[j]; col='#33ff77' if hit[j] and agree[j] else '#ffbf25'
        d.ellipse((u-1,vp-1,u+1,vp+1),fill=col)
    # Draw baseline/region points on primary overlay; write alternative coordinates separately.
    for name,entry in rec['coordinate_hypotheses'].items():
        p=np.asarray(entry['map_xyz_hypothesis']); pcam=(p-c2m[:3,3])@c2m[:3,:3]
        record={'camera_xyz_m':pcam,'positive_depth':bool(pcam[2]>0)}
        if pcam[2]>1e-6:
            qq=K@pcam; u,vp=qq[:2]/qq[2];inside=0<=u<w and 0<=vp<h
            record.update(pixel_uv=[u,vp],inside_image=bool(inside))
            if inside:
                ix,iy=int(math.floor(u)),int(math.floor(vp));dep=float(frame_depth[iy,ix])
                record.update(inside_mask=bool(mask[iy,ix]),rendered_depth_m=dep,
                              target_minus_visible_depth_m=float(pcam[2]-dep) if math.isfinite(dep) else None)
                if name in ('binding','region_center','region_surface'):
                    col=COLORS[name];d.line((u-6,vp,u+6,vp),fill=col,width=2);d.line((u,vp-6,u,vp+6),fill=col,width=2)
                    d.text((u+7,max(0,vp-12)),name,font=font(11),fill=col,stroke_width=1,stroke_fill='black')
        measurements['hypotheses'][name]=record
    d.rectangle((0,0,w,22),fill='#171d26');d.text((5,3),f'{rec["asset_id"]} | {v["frame_id"]} m{idx:04d} | green=mask+depth support',font=font(11),fill='white')
    image.save(out/f'v{index:02d}_projection.jpg',quality=92)
    yy,xx=np.where(mask)
    if len(xx):
        box=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(w,int(xx.max())+9),min(h,int(yy.max())+9))
        masked=rgb.copy();masked[~mask]=[238,234,241]
        Image.fromarray(masked).crop(box).save(out/f'v{index:02d}_crop.png')
    save_json(out/f'v{index:02d}_checks.json',measurements)
    return measurements


def cloud_projection(xyz, rgb, rec, dest, axes=(0,1)):
    if not len(xyz): return
    points=xyz
    if len(points)>60000:
        ids=np.linspace(0,len(points)-1,60000).astype(int);points=points[ids];rgb=rgb[ids]
    lo,hi=points[:,axes].min(0),points[:,axes].max(0)
    span=np.maximum(hi-lo,.10); pad=.08*max(span);lo-=pad;hi+=pad
    width=height=640; factor=min((width-60)/(hi[0]-lo[0]),(height-80)/(hi[1]-lo[1]))
    p=(points[:,axes]-lo)*factor
    a=Image.new('RGB',(width,height),'#222b34');d=ImageDraw.Draw(a)
    for uv,c in zip(p,rgb):
        x=30+uv[0];y=height-40-uv[1]
        d.ellipse((x-1,y-1,x+1,y+1),fill=tuple(map(int,c)))
    for key in ('binding','region_center','region_surface'):
        if key not in rec['coordinate_hypotheses']:continue
        xy=(np.array(rec['coordinate_hypotheses'][key]['map_xyz_hypothesis'])[list(axes)]-lo)*factor
        x,y=30+xy[0],height-40-xy[1]
        if 0<=x<width and 0<=y<height:
            col=COLORS[key];d.ellipse((x-5,y-5,x+5,y+5),outline=col,width=2)
            d.text((x+7,y-10),key,font=font(11),fill=col)
    d.text((10,10),rec['asset_id']+' | '+''.join('XYZ'[i] for i in axes)+' map projection',font=font(16),fill='white')
    a.save(dest)


def make_html(out, records, inspected):
    esc=lambda s:html.escape(str(s))
    rows=[]
    for r in records:
        aid=r['asset_id'];link=f'<a href="{esc(aid)}/index.html">Inspect</a>' if aid in inspected else 'Audit only'
        delta=r['binding_center_difference_m'];delta='—' if delta is None else f'{delta:.4f}'
        rows.append(f'<tr><td>{esc(aid)}</td><td>{esc(r["tag"])}</td><td>{esc(r["asset_class"])}</td><td>{esc(r["region_id"])}</td><td>{delta}</td><td>{link}</td></tr>')
    style='body{font:15px system-ui;margin:28px;background:#EEEAF1;color:#29263E}a{color:#4E43A4}table{border-collapse:collapse;background:#FFFCFF}td,th{padding:9px;border:1px solid #D8D1DC}img{max-width:95%;height:auto}pre{white-space:pre-wrap}section{padding:16px;background:#FFFCFF;margin:12px 0;border-radius:12px}'
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Read-only asset inspection</title><style>{style}</style><h1>Asset and coordinate inspection</h1><p>No web-demo files or mappings have been changed. IDs are current demo bindings, not verified physical asset identities.</p><p><a href="audit.json">Audit JSON</a> · <a href="assets.csv">CSV</a></p><img src="assets_current_map.png"><table><tr><th>Asset</th><th>Tag</th><th>Workbook class</th><th>Bound region</th><th>Target–center distance (m)</th><th>Evidence</th></tr>{"".join(rows)}</table><h2>Workbook anchors (placeholders)</h2><img src="workbook_placeholder_map.png">',encoding='utf-8')
    for r in records:
        aid=r['asset_id']
        if aid not in inspected:continue
        folder=out/aid;projections=sorted(folder.glob('v*_projection.jpg'))
        views=''.join(f'<section><h3>{esc(p.stem)}</h3><img style="max-width:24%" src="{p.stem.replace("projection","crop")}.png"><img style="max-width:72%" src="{p.name}"></section>' for p in projections)
        hypotheses=''.join(f'<tr><td>{esc(k)}</td><td>{esc(np.round(v["map_xyz_hypothesis"],4).tolist())}</td><td>{esc(v["map"]["state"])}</td><td>{esc(v["description"])}</td></tr>' for k,v in r['coordinate_hypotheses'].items())
        (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{esc(aid)}</title><style>{style}</style><a href="../index.html">All assets</a><h1>{esc(aid)} · {esc(r["tag"])}</h1><p>Workbook class: {esc(r["asset_class"])}. Bound region: {esc(r["region_id"])}. This association is not validated identity.</p><pre>{esc(chr(10).join(r["warnings"]))}</pre><p>Red=stored target, gold=current bbox center, teal=observed surface point. A center can be inside an object and behind its visible surface.</p><img style="max-width:48%" src="cloud_XY.png"><img style="max-width:48%" src="cloud_XZ.png"><h2>Coordinate hypotheses—NOT fixes</h2><table>{hypotheses}</table><h2>Original supporting views</h2>{views}<p><a href="asset.json">Complete numerical report</a></p>',encoding='utf-8')


def run_audit(args, inspect=False):
    s=Sources(args);out=Path(args.out).expanduser().resolve()
    roots=[s.bundle,s.region_root,s.map.path.parent]
    if s.enh_root:roots.append(s.enh_root)
    if any(out==p or p in out.parents or out in p.parents for p in roots):
        raise ValueError('Inspection output must be separate from all source folders.')
    if out.exists() and any(out.iterdir()): raise ValueError('Use a NEW empty output directory; existing reports are not overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    keys=sorted(s.db['assets']); selected=keys if args.assets.lower()=='all' else [a.strip() for a in args.assets.split(',') if a.strip()]
    invalid=set(selected)-set(keys)
    if invalid:raise ValueError('Unknown asset IDs: '+', '.join(invalid))
    records=[]; inspected=set(); env=None
    if inspect and s.env_path: env=s.environment()
    print(f'Inspecting {len(keys)} assets. Selection for geometry: {len(selected) if inspect else 0}',flush=True)
    checks=source_global_checks(s)
    for aid in keys:
        do=inspect and aid in selected
        rec,xyz,rgb=audit_record(s,aid,with_cloud=do);records.append(rec)
        print(f'{aid}: {rec["binding"].get("status")} | region={rec["region_id"]} | target-center={rec["binding_center_difference_m"]}',flush=True)
        if not do:continue
        inspected.add(aid);folder=out/aid;folder.mkdir();rid=rec['region_id']
        if xyz is not None:
            save_cloud(folder,'bound_object_map',xyz,rgb)
            cloud_projection(xyz,rgb,rec,folder/'cloud_XY.png',(0,1))
            cloud_projection(xyz,rgb,rec,folder/'cloud_XZ.png',(0,2))
            _,raw,en=s.region(rid)
            gp=en/'gaussians_original.ply' if en else Path('__missing__')
            if gp.exists():
                rec['original_gaussian_source']=str(gp)
                if args.copy_gaussians:shutil.copy2(gp,folder/'bound_object_gaussians_original.ply')
                try:
                    from plyfile import PlyData
                    ga = PlyData.read(str(gp), mmap=True)['vertex']
                    original_centers = np.column_stack([ga[n] for n in ('x','y','z')]).astype(float)
                    transformed = scene_to_map(original_centers,s.scale,s.floor)
                    if en and (en/'gaussian_centers_map.npz').exists():
                        saved_centers,_ = load_cloud(en/'gaussian_centers_map.npz')
                        if saved_centers.shape == transformed.shape:
                            rec['original_to_saved_gaussian_centers_max_error_m'] = float(np.max(np.linalg.norm(saved_centers-transformed,axis=1))) if len(transformed) else 0.
                            if rec['original_to_saved_gaussian_centers_max_error_m'] > .01:
                                rec['warnings'].append('GAUSSIAN_CENTER_TRANSFORM_MISMATCH: inspect scale/floor/frame metadata.')
                        else:rec['warnings'].append('GAUSSIAN_CENTER_COUNT_MISMATCH: cannot compare rows.')
                except ImportError:
                    rec['warnings'].append('PLYFILE_UNAVAILABLE: original Gaussian transform check skipped.')
                except Exception as e:
                    rec['warnings'].append('GAUSSIAN_TRANSFORM_CHECK_FAILED: '+str(e))
            if en and (en/'gaussian_centers_map.npz').exists():
                gx,gr=load_cloud(en/'gaussian_centers_map.npz');save_cloud(folder,'gaussian_centers_map',gx,gr)
                from scipy.spatial import cKDTree
                dist=cKDTree(xyz).query(gx[::max(1,len(gx)//10000)],workers=2)[0] if len(gx) else np.array([])
                rec['gaussian_center_to_surface_median_m']=float(np.median(dist)) if len(dist) else None
            rec['view_checks']=[]
            for i,v in enumerate(rec['source_views'][:args.views]):
                try:
                    check = independent_view_check(s,rec,xyz,v,folder,i)
                    rec['view_checks'].append(check)
                    if (check.get('camera_translation_discrepancy_m',0)>.01 or
                        check.get('camera_rotation_discrepancy',0)>1e-4 or
                        check.get('scale_matches_dashboard') is False or check.get('floor_matches_dashboard') is False):
                        rec['warnings'].append('CAMERA_TRANSFORM_DISAGREEMENT: '+v['frame_id'])
                except Exception as e:rec['view_checks'].append({'frame_id':v.get('frame_id'),'error':f'{type(e).__name__}: {e}'})
        if env:
            ex,ec=env
            for name,choice in rec['coordinate_hypotheses'].items():
                center=np.asarray(choice['map_xyz_hypothesis']);lo=center-args.radius;hi=center+args.radius
                k=np.all((ex>=lo)&(ex<=hi),axis=1)
                if int(k.sum())>args.max_context:
                    ids=np.flatnonzero(k);ids=ids[np.linspace(0,len(ids)-1,args.max_context).astype(int)]
                    cx,cc=ex[ids],ec[ids]
                else:cx,cc=ex[k],ec[k]
                choice['context_points_before_cap']=int(k.sum())
                # A local spatial crop is not semantic extraction.
                save_cloud(folder,'context_'+name,cx,cc)
        save_json(folder/'asset.json',rec)
    export_overview(s,records,out,checks)
    make_html(out,records,inspected)
    preserved=s.unchanged()
    summary={'asset_count':len(records),'geometrically_inspected':len(inspected),
        'procedural_bindings':sum(r['binding'].get('status')=='procedural_demo_binding' for r in records),
        'unmapped':sum(vec3(r['binding'].get('map_xyz')) is None for r in records),
        'target_region_center_difference_gt_10cm':sum((r['binding_center_difference_m'] or 0)>.10 for r in records),
        'global_failed_checks':[c['check'] for c in checks if not c['pass']],
        'source_files_unchanged':preserved,'monitored_source_sha256':s.monitor,
        'scope':'Source/code diagnosis only; no identity/registration automatically corrected.'}
    save_json(out/'summary.json',summary)
    print(json.dumps(jsonable(summary),indent=2));print('Report:',out/'index.html')
    if not preserved: raise RuntimeError('A monitored source changed while inspecting; stop upstream writers and use a new report.')


def view_report(args):
    try:import open3d as o3d
    except ImportError:raise RuntimeError('Open3D is not installed in this environment. Use the generated HTML/PNGs instead; no install is needed for inspection.')
    folder=Path(args.out).resolve()/args.asset;r=load_json(folder/'asset.json')
    geoms=[]
    context=folder/('context_'+args.mode+'.ply')
    if context.exists():
        cloud=o3d.io.read_point_cloud(str(context));
        if len(cloud.points):
            cloud.colors=o3d.utility.Vector3dVector(np.asarray(cloud.colors)*.45+.2);geoms.append(cloud)
    p=folder/'bound_object_map.ply'
    if p.exists():
        cloud=o3d.io.read_point_cloud(str(p));geoms.append(cloud)
        if len(cloud.points):
            bb=cloud.get_axis_aligned_bounding_box();bb.color=[.1,.85,.5];geoms.append(bb)
    choice=r['coordinate_hypotheses'].get(args.mode)
    if not choice:raise ValueError('No coordinate for this mode')
    for key,col in [(args.mode,[1.,.12,.1]),('region_center',[1.,.72,.1]),('region_surface',[.1,1.,.6])]:
        if key in r['coordinate_hypotheses']:
            sphere=o3d.geometry.TriangleMesh.create_sphere(radius=.05);sphere.translate(r['coordinate_hypotheses'][key]['map_xyz_hypothesis']);sphere.paint_uniform_color(col);sphere.compute_vertex_normals();geoms.append(sphere)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.4,origin=choice['map_xyz_hypothesis']))
    print('RED: chosen target hypothesis; GOLD: current box center; GREEN: retained surface point. All display coordinates are map metres (provisional).')
    o3d.visualization.draw_geometries(geoms,window_name=f'Read-only: {args.asset} / {args.mode}',width=1200,height=800)


def render_report(args):
    # Lazy import: audit/inspect do not initialize CUDA or change CUDA variables.
    folder=Path(args.out).resolve()/args.asset;rec=load_json(folder/'asset.json')
    audit=load_json(Path(args.out)/'audit.json');cfg=audit['config'];s=float(cfg['scale']);floor=float(cfg['floor_y'])
    target=rec['coordinate_hypotheses'].get(args.mode)
    if target is None:raise ValueError('Target hypothesis unavailable')
    p=np.asarray(target['map_xyz_hypothesis'],float)
    if args.view_index>=len(rec['source_views']):raise ValueError('Source view index unavailable')
    v=rec['source_views'][args.view_index];cam=load_json(Path(v['source_frame'])/'camera.json')
    # Use actual original source camera unless the user explicitly requests new aiming.
    K=np.asarray(cam['K'],np.float32);c2g=np.asarray(cam['camera_to_world'],np.float32)
    w,h=int(cam['width']),int(cam['height'])
    ratio=min(1.,args.width/w);w=round(w*ratio);h=round(h*ratio);K[:2,:]*=ratio
    if args.aim_at_target:
        c2m=camera_to_map(c2g,s,floor);delta=p-c2m[:3,3];d=float(np.linalg.norm(delta))
        if d<1e-6:raise ValueError('Camera and target coincide')
        from avatar_demo.core import camera_matrices
        yaw=math.atan2(delta[1],delta[0]);pitch=float(np.clip(math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2]))),-80,80))
        # Preserve original focal calibration; camera_matrices provides orientation only.
        _,c2g,_=camera_matrices(c2m[0,3],c2m[1,3],yaw,c2m[2,3],pitch,scale=s,floor_y=floor,width=w,height_px=h)
    import torch
    from gsplat import rasterization
    from avatar_demo.render_worker import _load_scene
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run the known driver diagnostic. No packages changed.')
    scene=args.scene or cfg['ply']; print('Loading original Gaussian room, read-only...',flush=True)
    means,quats,scales,opacity,sh=_load_scene(scene,torch,3)
    with torch.inference_mode():
        pixels,alphas,_=rasterization(means=means,quats=quats,scales=scales,opacities=opacity,colors=sh,
             viewmats=torch.from_numpy(np.linalg.inv(c2g).astype(np.float32)[None]).cuda(),
             Ks=torch.from_numpy(K[None]).cuda(),width=w,height=h,sh_degree=3,packed=True,
             backgrounds=None,render_mode='RGB+ED',near_plane=.03/s,far_plane=60/s)
        arr=pixels[0].float().cpu().numpy();alpha=alphas[0,:,:,0].float().cpu().numpy()
    rgb=np.rint(np.clip(arr[:,:,:3],0,1)*255).astype(np.uint8)
    stem=f'render_{args.mode}_v{args.view_index:02d}'+('_aimed' if args.aim_at_target else '_source')
    if (folder/(stem+'_rgb.png')).exists():raise ValueError('This diagnostic render already exists; use another output report or view index.')
    Image.fromarray(rgb).save(folder/(stem+'_rgb.png'));np.save(folder/(stem+'_depth_scene.npy'),arr[:,:,3]);np.save(folder/(stem+'_alpha.npy'),alpha)
    c2m=camera_to_map(c2g,s,floor);pp=(p-c2m[:3,3])@c2m[:3,:3];proj=None
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    if pp[2]>0:
        q=K@pp;u,vp=q[:2]/q[2];proj=[float(u),float(vp)]
        if 0<=u<w and 0<=vp<h:
            draw.line((u-10,vp,u+10,vp),fill='red',width=2);draw.line((u,vp-10,u,vp+10),fill='red',width=2)
    draw.text((6,6),args.asset+' | target projection, not recognition',font=font(14),fill='white',stroke_width=1,stroke_fill='black')
    im.save(folder/(stem+'_target.png'))
    save_json(folder/(stem+'_camera.json'),{'K':K,'camera_to_world':c2g,'camera_to_map':c2m,'width':w,'height':h,
              'target_map':p,'target_pixel':proj,'hypothesis':args.mode,'depth_units':'original Gaussian scene units',
              'note':'Rendered point projection can be occluded or inside object. No target recognition or service-demo changes.'})
    print('Saved:',folder/(stem+'_target.png'))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    for action in ('audit','inspect'):
        a=sp.add_parser(action);a.add_argument('--bundle',default='outputs/service_demo/main_v1')
        a.add_argument('--out',required=True);a.add_argument('--assets',default='all',help='all or comma-separated ASSET-001,ASSET-002')
        a.add_argument('--regions');a.add_argument('--enhancement');a.add_argument('--map');a.add_argument('--environment')
        a.add_argument('--views',type=int,default=3);a.add_argument('--radius',type=float,default=1.5,help='Context cube half-width in provisional metres')
        a.add_argument('--max-context',type=int,default=100000);a.add_argument('--copy-gaussians',action='store_true')
    a=sp.add_parser('view');a.add_argument('--out',required=True);a.add_argument('--asset',required=True);a.add_argument('--mode',choices=MODES,default='binding')
    a=sp.add_parser('render');a.add_argument('--out',required=True);a.add_argument('--asset',required=True)
    a.add_argument('--mode',choices=MODES,default='binding');a.add_argument('--view-index',type=int,default=0)
    a.add_argument('--width',type=int,default=640);a.add_argument('--scene');a.add_argument('--aim-at-target',action='store_true')
    return p


def main():
    args=parser().parse_args()
    if args.command in ('audit','inspect'):
        if args.radius<=0 or args.views<0 or args.max_context<1:raise ValueError('Invalid radius/views/context limit')
        run_audit(args,args.command=='inspect')
    elif args.command=='view':view_report(args)
    else:render_report(args)


if __name__=='__main__':
    # Ensure the existing project modules can be imported without installing them.
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    try:main()
    except KeyboardInterrupt:print('\nStopped. Original data unchanged.',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
