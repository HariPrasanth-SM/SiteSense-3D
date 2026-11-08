from __future__ import annotations
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw
from .geometry import box_corners, box_from_points, project, map_to_scene, voxel_cloud, transform_record


def now_id(): return time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]


def clean_json(obj):
    if isinstance(obj, np.ndarray): return clean_json(obj.tolist())
    if isinstance(obj, np.generic): return clean_json(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj): return None
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {str(k):clean_json(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean_json(v) for v in obj]
    return obj


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd,'w') as f: json.dump(clean_json(obj), f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for data in iter(lambda:f.read(1024*1024), b''): h.update(data)
    return h.hexdigest()


def image_data(a, fmt='PNG'):
    b=io.BytesIO(); Image.fromarray(np.asarray(a, np.uint8)).save(b, format=fmt)
    return b.getvalue()


def depth_preview(frame, far=8):
    d = frame['depth_m']; valid=np.isfinite(d)&(d>0)&(frame['alpha']>=.5)
    val=np.zeros(d.shape,np.uint8); val[valid]=np.rint(255*(1-np.clip(d[valid]/far,0,1))).astype(np.uint8)
    return val


def write_ply(path, xyz, rgb):
    """Binary ordinary XYZRGB surface samples. This is NOT a Gaussian-splat file."""
    points=np.empty(len(xyz), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(('x','y','z')): points[k]=xyz[:,i]
    for i,k in enumerate(('red','green','blue')): points[k]=rgb[:,i]
    with open(path,'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\n'
                'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
        f.write(points.tobytes())


def overlay(frame, draft):
    rgb=frame['rgb'].copy(); mask=draft['support_mask']
    rgb[mask]=np.rint(.62*rgb[mask]+.38*np.array([40,225,125])).astype(np.uint8)
    image=Image.fromarray(rgb); draw=ImageDraw.Draw(image)
    box=box_corners(draft['box']); uv,ok=project(box, frame['camera'])
    from .geometry import EDGES
    for a,b in EDGES:
        if ok[a] and ok[b] and np.abs(uv[[a,b]]).max()<1e6:
            draw.line([tuple(uv[a]),tuple(uv[b])],fill=(250,186,60),width=2)
    p,valid=project(np.asarray([draft['target']]),frame['camera'])
    if valid[0] and np.abs(p).max()<1e6:
        x,y=p[0];draw.ellipse([x-5,y-5,x+5,y+5],fill=(220,50,80))
    return image


class AnnotationStore:
    def __init__(self, out, identity):
        self.root=Path(out).expanduser().resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        cfg=self.root/'source.json'
        if cfg.exists():
            previous=json.loads(cfg.read_text())
            for k in ('scene_sha256','map_yaml_sha256','map_image_sha256','scale','floor_y'):
                if previous.get(k)!=identity.get(k):
                    raise ValueError(f'Annotation output belongs to different source/transform ({k}). Use a NEW --out folder.')
        else: atomic_json(cfg,identity)
        self.objects={}
        for path in sorted((self.root/'objects').glob('manual_*/object.json')):
            rec=json.loads(path.read_text());self.objects[rec['object_id']]=rec
        self.export_catalog()

    def export_catalog(self):
        entries=list(self.objects.values())
        catalog={'schema':'sitesense.manual_labels.v1','created_at':time.time(),
                 'source':self.identity,'frame':'map','units':'provisional_metres',
                 'objects':entries,
                 'notes':['Human-drawn 2D outlines with expected Gaussian depth; partial surfaces, not complete true object volumes.',
                          'Labels are manually supplied. No Excel assets or automatic synthetic bindings.',
                          'Object aim point is not a navigation stopping point. Reference camera poses are not guaranteed reachable.']}
        atomic_json(self.root/'annotations.json',catalog)
        dst=self.root/'annotations.csv';tmp=dst.with_suffix('.csv.tmp')
        with open(tmp,'w',newline='') as f:
            fields=['object_id','label','category','active','view_count','target_x','target_y','target_z','center_x','center_y','center_z','size_x','size_y','size_z','yaw_deg','cloud_path','note']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for o in entries:
                b=o['bbox_map'];r=dict(object_id=o['object_id'],label=o['label'],category=o['category'],active=o['active'],view_count=len(o['observations']),yaw_deg=b.get('yaw_deg',0),cloud_path=o['cloud_path'],note=o['note'])
                for name,v in zip(('target_x','target_y','target_z'),o['target_map_xyz']):r[name]=v
                for name,v in zip(('center_x','center_y','center_z'),b['center']):r[name]=v
                for name,v in zip(('size_x','size_y','size_z'),b['extent']):r[name]=v
                w.writerow(r)
        os.replace(tmp,dst)

    def save(self,frame,draft,label,category,note,existing=None):
        label=str(label).strip();category=str(category).strip();note=str(note).strip()
        if not label or len(label)>160 or len(category)>80 or len(note)>4000:
            raise ValueError('Supply a short label (1–160 characters), category <=80, note <=4000.')
        if existing:
            if existing not in self.objects: raise ValueError('Saved object not found.')
            previous=self.objects[existing];oid=existing
        else:
            next_id=max([int(x.rsplit('_',1)[1]) for x in self.objects]+[0])+1
            oid=f'manual_{next_id:04d}';previous=None
        root=self.root/'objects'/oid;obs_id='view_'+now_id();d=root/'observations'/obs_id;d.mkdir(parents=True)
        camera=frame['camera']
        Image.fromarray(frame['rgb']).save(d/'rgb.png')
        np.save(d/'depth_scene.npy',frame['depth_scene'].astype(np.float32))
        np.save(d/'depth_m.npy',frame['depth_m'].astype(np.float32))
        np.save(d/'alpha.npy',frame['alpha'].astype(np.float32))
        atomic_json(d/'camera.json',camera)
        Image.fromarray(draft['raw_mask'].astype(np.uint8)*255).save(d/'mask_drawn.png')
        Image.fromarray(draft['support_mask'].astype(np.uint8)*255).save(d/'mask_depth_supported.png')
        overlay(frame,draft).save(d/'overlay.png')
        vv,uu=np.nonzero(draft['raw_mask']);bounds=[int(uu.min()),int(vv.min()),int(uu.max()+1),int(vv.max()+1)]
        rgba=np.dstack([frame['rgb'],draft['support_mask'].astype(np.uint8)*255])
        Image.fromarray(rgba).crop(bounds).save(d/'object_crop.png')
        np.savez_compressed(d/'samples_map.npz',xyz=draft['xyz'],rgb=draft['rgb'],uv=draft['uv'],
                            xyz_before_sor=draft['xyz_before_sor'],uv_before_sor=draft['uv_before_sor'],
                            retained_indices=draft['retained_samples'])
        write_ply(d/'surface_map.ply',draft['xyz'],draft['rgb'])
        rel=lambda p:str(p.relative_to(self.root))
        observation={'observation_id':obs_id,'object_id':oid,'frame_id':frame['id'],
                     'frozen_frame_pose':camera['map_pose'],'frame_captured_at':camera.get('captured_wall_time'),
                     'camera':rel(d/'camera.json'),'rgb':rel(d/'rgb.png'),'overlay':rel(d/'overlay.png'),
                     'crop':rel(d/'object_crop.png'),'samples':rel(d/'samples_map.npz'),
                     'shape':draft['shape'],'depth_selection':draft['settings'],'stats':draft['stats'],
                     'target_map_xyz':draft['target'],'aim_source':draft['aim_source'],
                     'view_box':draft['box'],'box_was_human_adjusted':draft['box_manual'],
                     'source_coordinate_transform':transform_record(camera['scale'],camera['floor_y']).tolist(),
                     'warnings':draft['warnings']}
        atomic_json(d/'observation.json',observation)
        hashes={p.name:sha(p) for p in d.iterdir() if p.is_file()}
        atomic_json(d/'complete.json',{'files':hashes})
        observations=(previous['observations'][:] if previous else [])+[{'id':obs_id,'path':rel(d/'observation.json')}]
        xyz=draft['xyz'];rgb=draft['rgb']
        if previous:
            with np.load(self.root/previous['cloud_npz'],allow_pickle=False) as old:
                xyz=np.vstack([old['xyz'],xyz]);rgb=np.vstack([old['rgb'],rgb])
        xyz,rgb=voxel_cloud(xyz,rgb,.01)
        observed=box_from_points(xyz)
        if draft['box_manual']:box=draft['box'];box_policy='human_adjusted_box'
        elif previous and previous.get('box_policy')=='human_adjusted_box':box=previous['bbox_map'];box_policy=previous['box_policy']
        else:box=observed;box_policy='bounds_of_accumulated_observed_surfaces'
        revision=1+(previous.get('revision',0) if previous else 0)
        rdir=root/'revisions'/f'rev{revision:04d}';rdir.mkdir(parents=True)
        np.savez_compressed(rdir/'cloud_map.npz',xyz=xyz,rgb=rgb)
        write_ply(rdir/'cloud_map.ply',xyz,rgb)
        write_ply(rdir/'cloud_scene.ply',map_to_scene(xyz,camera['scale'],camera['floor_y']),rgb)
        corners=box_corners(box)
        rec={'object_id':oid,'label':label,'category':category,'note':note,'active':True,'revision':revision,
             'created_at':previous['created_at'] if previous else time.time(),'updated_at':time.time(),
             'target_map_xyz':draft['target'],'target_scene_xyz':map_to_scene(draft['target'],camera['scale'],camera['floor_y']).tolist(),
             'aim_source':draft['aim_source'],'bbox_map':box,'bbox_corners_map':corners.tolist(),
             'bbox_corners_scene':map_to_scene(corners,camera['scale'],camera['floor_y']).tolist(),
             'observed_surface_bbox_map':observed,'box_policy':box_policy,
             'cloud_path':rel(rdir/'cloud_map.ply'),'cloud_npz':rel(rdir/'cloud_map.npz'),
             'cloud_scene_path':rel(rdir/'cloud_scene.ply'),'point_count':len(xyz),'observations':observations,
             'reference_crop':rel(d/'object_crop.png'),'reference_overlay':rel(d/'overlay.png'),
             'reference_camera':rel(d/'camera.json'),'reference_camera_pose':camera['map_pose'],
             'verified_navigation_goal':None,'semantic_source':'manual_user_label','coordinate_frame':'map',
             'geometry_note':'Partial rendered-depth surface samples. Bbox edits are recorded human estimates; neither is verified physical extent.'}
        atomic_json(rdir/'object.json',rec);atomic_json(root/'object.json',rec)
        self.objects[oid]=rec;self.export_catalog();return rec

    def rename(self,oid,label,category,note,active):
        if oid not in self.objects:raise ValueError('Object not found.')
        if not str(label).strip() or len(str(label))>160:raise ValueError('Label must contain 1–160 characters.')
        rec=dict(self.objects[oid]);rec.update(label=str(label).strip(),category=str(category)[:80],note=str(note)[:4000],active=bool(active),updated_at=time.time())
        atomic_json(self.root/'objects'/oid/('review_'+now_id()+'.json'),rec)
        atomic_json(self.root/'objects'/oid/'object.json',rec);self.objects[oid]=rec;self.export_catalog();return rec
