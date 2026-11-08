from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import json
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from semantic_scene.common import (read_json,write_json,write_npz,digest,sha,checked,
                                   file_records,load_scene,lock)
from semantic_scene.fusion import bounding_box
from .geometry import statistical_keep,write_rgb_ply,source_mask,scene_to_map,view_vote


class FrameCache:
    def __init__(self):self.frames=OrderedDict()
    def get(self,view):
        folder=Path(view['source_frame'])
        key=str(folder.resolve())
        if key in self.frames:self.frames.move_to_end(key);return self.frames[key]
        with Image.open(folder/'rgb.png') as im:rgb=np.array(im.convert('RGB'))
        depth=np.load(folder/'depth_m.npy',allow_pickle=False)
        alpha=np.load(folder/'alpha.npy',allow_pickle=False)
        camera=read_json(folder/'camera.json')
        if rgb.shape[:2]!=depth.shape or alpha.shape!=depth.shape:raise ValueError(f'Source shape mismatch: {folder}')
        data={'rgb':rgb,'depth':depth,'alpha':alpha,'camera':camera}
        self.frames[key]=data
        while len(self.frames)>4:self.frames.popitem(last=False)
        return data


def finite_params(v):
    fields=['x','y','z','opacity']+[f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    fields += [n for n in v.dtype.names if n.startswith(('f_dc_','f_rest_'))]
    ok=np.ones(len(v),bool)
    for name in fields:ok &= np.isfinite(v[name])
    norm=sum(np.asarray(v[f'rot_{i}'],float)**2 for i in range(4))
    return ok & (norm>1e-16)


def prepare(args):
    from plyfile import PlyData,PlyElement
    src,scene=load_scene(args.scene)
    out=args.out.expanduser().resolve()
    # Keep enrichment entirely separate. Labels remain in the original scene.
    if out==src or src in out.parents or out in src.parents:
        raise ValueError('Use --out separate from the original scene (e.g. outputs/semantic_enhance/...)')
    ply_path=args.gaussian_ply.expanduser().resolve() if args.gaussian_ply else None
    config={'schema':'valmet.semantic_enhance.v1','version':'1.0.0','source_scene':str(src),
            'source_scene_id':scene['scene_id'],'source_scene_sha256':sha(src/'scene.json'),
            'gaussian_ply':str(ply_path) if ply_path else None,
            'gaussian_sha256':sha(ply_path) if ply_path else None,
            'scale':float(scene['scale_m_per_scene_unit']),'floor_y':float(scene.get('floor_y_gaussian',1.1)),
            'neighbors':args.neighbors,'std_ratio':args.std_ratio,'splat_radius_m':args.splat_radius,
            'min_splat_views':args.min_splat_views,'depth_abs_m':args.depth_abs,'depth_rel':args.depth_rel,
            'max_gaussian_sigma_m':args.max_gaussian_sigma,'mask_dilation_pixels':1,'alpha_min':.5}
    config['enhancement_id']=digest(config)
    with lock(out):
        if (out/'config.json').exists() and read_json(out/'config.json')!=config:
            raise ValueError('Settings/source differ. Use a NEW --out directory; no overwrite performed.')
        write_json(out/'config.json',config)
        v=None;all_map=None;good=None;original_comments=[]
        if ply_path:
            print('Loading original Gaussian records on CPU (read-only)...',flush=True)
            ply=PlyData.read(str(ply_path),mmap=True);v=ply['vertex'].data
            required={'x','y','z','opacity'}|{f'scale_{i}' for i in range(3)}|{f'rot_{i}' for i in range(4)}|{f'f_dc_{i}' for i in range(3)}
            if not required.issubset(v.dtype.names):raise ValueError('Pass the ORIGINAL Gaussian PLY, not an XYZRGB export')
            good=finite_params(v)
            if args.max_gaussian_sigma:
                logs=np.column_stack([v[f'scale_{i}'] for i in range(3)])
                good &= np.exp(np.clip(logs,-30,20)).max(1)*config['scale']<=args.max_gaussian_sigma
            all_map=scene_to_map(np.column_stack([v[n] for n in ('x','y','z')]),config['scale'],config['floor_y']).astype(np.float32)
            original_comments=list(ply.comments)
        frames=FrameCache();records=[];new=0;started=time.perf_counter()
        selected=scene['regions']
        if args.region:
            selected=[r for r in selected if r['region_id']==args.region]
            if not selected:raise ValueError('Unknown --region ID')
        for i,item in enumerate(selected,1):
            rid=item['region_id'];raw=src/'regions'/rid;folder=out/'regions'/rid
            if Path(rid).name!=rid:raise ValueError('Invalid region ID')
            region=read_json(raw/'region.json')
            stamp=read_json(raw/'complete.json')
            checked(raw/'surface.npz',stamp['files']['surface.npz'])
            checked(raw/'region.json',stamp['files']['region.json'])
            source_identity={'region_sha256':sha(raw/'region.json'),'surface_sha256':stamp['files']['surface.npz']['sha256']}
            done=folder/'complete.json'
            if done.exists():
                d=read_json(done)
                if d['enhancement_id']!=config['enhancement_id'] or d['source']!=source_identity:raise ValueError('Changed region source; new output required')
                for name,entry in d['files'].items():checked(folder/name,entry)
                print(f'[{i}/{len(selected)}] reused {rid}',flush=True);continue
            if args.limit is not None and new>=args.limit:break
            folder.mkdir(parents=True,exist_ok=True);(folder/'views').mkdir(exist_ok=True)
            with np.load(raw/'surface.npz',allow_pickle=False) as a:
                xyz=a['xyz'].astype(np.float32);rgb=a['rgb'].astype(np.uint8)
                station=a['station_count'].copy();obs_count=a['observation_count'].copy() if 'observation_count' in a else station.copy()
                offsets=a['membership_offsets'].copy() if 'membership_offsets' in a else None
                member=a['membership_observation_indices'].copy() if 'membership_observation_indices' in a else None
            keep,stats=statistical_keep(xyz,args.neighbors,args.std_ratio)
            if int(keep.sum())<3:raise ValueError(f'{rid}: fewer than 3 points retained; review source before proceeding')
            ids=np.flatnonzero(keep)
            arrays={'xyz':xyz[keep],'rgb':rgb[keep],'source_surface_indices':ids,
                    'station_count':station[keep],'observation_count':obs_count[keep]}
            if offsets is not None:
                chunks=[member[offsets[j]:offsets[j+1]] for j in ids]
                arrays['membership_offsets']=np.r_[0,np.cumsum([len(c) for c in chunks])].astype(np.int64)
                arrays['membership_observation_indices']=np.concatenate(chunks) if chunks else np.zeros(0,np.int32)
            write_npz(folder/'surface_clean.npz',**arrays)
            write_npz(folder/'surface_removed.npz',xyz=xyz[~keep],rgb=rgb[~keep],source_surface_indices=np.flatnonzero(~keep))
            write_rgb_ply(folder/'surface_clean.ply',xyz[keep],rgb[keep])
            write_rgb_ply(folder/'surface_removed.ply',xyz[~keep],rgb[~keep])
            clean=xyz[keep];clean_bbox=bounding_box(clean)
            files=['surface_clean.npz','surface_clean.ply','surface_removed.npz','surface_removed.ply']
            # Center-based Gaussian proposal. Preserve original coordinates and ALL attributes on export.
            candidate_ids=np.zeros(0,np.int64);nearest=np.zeros(0,np.float32)
            if v is not None:
                lo=clean.min(0)-args.splat_radius;hi=clean.max(0)+args.splat_radius
                candidate_ids=np.flatnonzero(good & (all_map>=lo).all(1) & (all_map<=hi).all(1))
                if len(candidate_ids):
                    dist,_=cKDTree(clean).query(all_map[candidate_ids],k=1,workers=2)
                    near=dist<=args.splat_radius;candidate_ids=candidate_ids[near];nearest=dist[near].astype(np.float32)
            support=np.zeros((len(candidate_ids),len(region['views'])),bool)
            views=[];source_evidence={}
            for j,view in enumerate(region['views']):
                f=frames.get(view);mask=source_mask(Path(view['mask_archive']),int(view['mask_index']))
                if mask.shape!=f['rgb'].shape[:2]:raise ValueError('Mask is not aligned to RGB')
                if not np.allclose(view['camera_to_map'],f['camera']['camera_to_map'],atol=1e-6):raise ValueError('Source pose mismatch')
                if not np.allclose(view['K'],f['camera']['K'],atol=1e-6):raise ValueError('Source intrinsics mismatch')
                overlay=f['rgb'].copy();col=np.asarray(region['color'],float)
                overlay[mask]=np.clip(.68*overlay[mask]+.32*col,0,255).astype(np.uint8)
                edge=mask & ~binary_erosion(mask);overlay[edge]=[255,220,60]
                name=f'views/v{j:02d}_overlay.jpg';Image.fromarray(overlay).save(folder/name,quality=94)
                files.append(name)
                views.append({'local_index':j,'frame_id':view['frame_id'],'mask_index':view['mask_index'],
                              'station_id':view['station_id'],'overlay':name})
                if len(candidate_ids):
                    for first in range(0,len(candidate_ids),100000):
                        sl=slice(first,first+100000)
                        support[sl,j]=view_vote(all_map[candidate_ids[sl]],mask,f['depth'],f['alpha'],
                            f['camera']['K'],f['camera']['camera_to_map'],args.depth_abs,args.depth_rel)
                # Completion stamps link original evidence without duplicating it.
                source_dir=Path(view['source_frame']);seg_dir=Path(view['mask_archive']).parent
                for p in (source_dir/'complete.json',seg_dir/'complete.json'):
                    if p.exists():source_evidence[str(p)]=sha(p)
            # Count distinct station IDs, not duplicated masks in the same image/camera.
            station_ids=list(dict.fromkeys(q['station_id'] for q in region['views']))
            count=np.zeros(len(candidate_ids),np.int16)
            for sid in station_ids:
                columns=[j for j,q in enumerate(region['views']) if q['station_id']==sid]
                count += support[:,columns].any(1)
            accepted=count>=args.min_splat_views
            g_ids=candidate_ids[accepted]
            g_xyz=all_map[g_ids] if v is not None else np.zeros((0,3),np.float32)
            if v is not None:
                dc=np.column_stack([v[f'f_dc_{j}'][g_ids] for j in range(3)])
                colors=np.rint(np.clip(.5+.28209479177387814*dc,0,1)*255).astype(np.uint8)
                write_npz(folder/'gaussian_centers_map.npz',xyz=g_xyz,rgb=colors,
                          original_gaussian_indices=g_ids,supporting_station_count=count[accepted],
                          supporting_view_mask=support[accepted],nearest_clean_distance_m=nearest[accepted])
                np.save(folder/'gaussian_indices.npy',g_ids)
                files+=['gaussian_centers_map.npz','gaussian_indices.npy']
                if len(g_ids):
                    # Never transform the PLY XYZ alone: covariance/SH orientation would then be inconsistent.
                    subset=v[g_ids].copy()
                    PlyData([PlyElement.describe(subset,'vertex')],text=False,byte_order='<',
                            comments=original_comments+['Valmet region subset; ORIGINAL Gaussian scene coordinates; nonexclusive heuristic selection']).write(str(folder/'gaussians_original.ply'))
                    files.append('gaussians_original.ply')
            note=[]
            if stats.get('removed',0)>.2*len(xyz):note.append('SOR removed >20%: inspect thin parts / density variation')
            if v is not None and not len(g_ids):note.append('No splats passed current support rule. No automatic fallback or invented splats.')
            record={'region_id':rid,'enhancement_id':config['enhancement_id'],'source':source_identity,
                    'cleaning':stats,'clean_bbox':clean_bbox,'raw_bbox':region['bbox'],
                    'gaussian_bbox':bounding_box(g_xyz) if len(g_xyz) else None,
                    'candidate_gaussians':len(candidate_ids),'selected_gaussians':len(g_ids),
                    'gaussian_min_station_support':args.min_splat_views,
                    'views':views,'source_evidence_stamps':source_evidence,'warnings':note,
                    'notes':['Independent subsets may overlap. No exclusive point ownership.',
                       'SOR removes sparse outliers, not incorrect dense attachments or incorrect group identity.',
                       'Point clouds/boxes: map frame. Gaussian PLY: ORIGINAL Gaussian frame; all original properties retained.',
                       'Gaussian membership is center-near-surface plus mask/depth support, NOT exact rasterizer contribution tracing.',
                       'Single-view/occluded/back surfaces may be absent; masks and expected depth may be inaccurate.']}
            write_json(folder/'enhancement.json',record);files.append('enhancement.json')
            write_json(done,{'enhancement_id':config['enhancement_id'],'source':source_identity,'files':file_records(folder,files)})
            new+=1
            print(f'[{i}/{len(selected)}] {rid}: {len(xyz):,} -> {len(clean):,} points | {len(g_ids):,} original splats | {len(views)} evidence views',flush=True)
        # Include all previously completed regions on resuming or preparing a different single region.
        for item in scene['regions']:
            p=out/'regions'/item['region_id']/'complete.json'
            if p.exists():
                d=read_json(p)
                if d['enhancement_id']==config['enhancement_id']:
                    records.append(read_json(p.parent/'enhancement.json'))
        summary={'enhancement_id':config['enhancement_id'],'source_scene':str(src),
            'source_regions':len(scene['regions']),'completed_regions':len(records),'new_this_run':new,
            'removed_points':sum(r['cleaning'].get('removed',0) for r in records),
            'selected_gaussians_sum_with_overlap':sum(r['selected_gaussians'] for r in records),
            'seconds_this_run':time.perf_counter()-started,'regions':[r['region_id'] for r in records]}
        write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
