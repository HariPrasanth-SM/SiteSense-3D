from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import csv
import json
import time
import numpy as np
from PIL import Image
from .common import (read_json,write_json,write_npz,digest,sha,checked,file_records,
                     color_for,code_hash,unit)
from .fusion import fuse_samples,bounding_box,select_members,distinct_positions


def crop_pair(rgb,mask,size=384):
    """Original, unannotated masked crop plus matched-context crop; aspect preserved."""
    mask=np.asarray(mask,bool);rgb=np.asarray(rgb,np.uint8)
    if mask.shape!=rgb.shape[:2] or rgb.shape[2]!=3:raise ValueError('RGB/mask shape mismatch')
    yy,xx=np.nonzero(mask)
    if not len(xx):raise ValueError('Empty kept mask')
    pad=max(4,int(.12*max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)))
    x0=max(0,int(xx.min())-pad);x1=min(mask.shape[1],int(xx.max())+pad+1)
    y0=max(0,int(yy.min())-pad);y1=min(mask.shape[0],int(yy.max())+pad+1)
    patch=rgb[y0:y1,x0:x1].copy();masked=patch.copy();masked[~mask[y0:y1,x0:x1]]=[124,116,104]
    def square(a):
        im=Image.fromarray(a);im.thumbnail((size,size),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(size,size),(124,116,104));canvas.paste(im,((size-im.width)//2,(size-im.height)//2))
        return canvas
    return square(masked),square(patch),[x0,y0,x1,y1]


def build(args):
    from associate_dataset.catalog import Catalog
    from associate_dataset.geometry import GeometryStore,MatchPolicy
    from associate_dataset.common import export_ply
    assoc=Path(args.association).expanduser().resolve();out=Path(args.out).expanduser().resolve()
    run=assoc/'runs'/args.run
    catalog=Catalog(assoc)
    recipe=read_json(run/'run.json');summary=read_json(run/'summary.json')
    if recipe['catalog_id']!=catalog.id or summary['run_id']!=recipe['run_id']:
        raise ValueError('Association run/catalog mismatch')
    source_paths=[assoc,*[Path(catalog.source[k]) for k in ('dataset','segmentation','features')]]
    if any(out==p or p in out.parents or out in p.parents for p in source_paths):
        raise ValueError('Use a new output directory separate from every source')
    records=[(p,read_json(p)) for p in sorted((run/'queries').glob('*/result.json'))]
    if len(records)!=summary['query_count']:raise ValueError('Run is incomplete; finish the association pilot first')
    config={'association':str(assoc),'run':args.run,'run_id':recipe['run_id'],'catalog_id':catalog.id,
            'source':catalog.source,'source_run_sha256':sha(run/'run.json'),
            'query_result_sha256':{p.parent.name:sha(p) for p,_ in records},
            'samples_per_view':args.samples,'max_views':args.max_views,'voxel_m':args.voxel,
            'min_views':args.min_views,'code_sha256':code_hash()}
    config['scene_id']=digest(config)
    if (out/'build.json').exists() and read_json(out/'build.json')!=config:
        raise ValueError('Output recipe differs. Use a NEW --out, never overwrite an existing reviewed scene.')
    write_json(out/'build.json',config)
    policy=replace(MatchPolicy(**recipe['policy']),samples=args.samples).validate()
    store=GeometryStore(catalog,policy,max_frames=5,max_surfaces=12)
    regions=[];omitted=[];started=time.perf_counter()
    seen_ids=set()
    for number,(path,record) in enumerate(records,1):
        q=int(record['query_index']);members=select_members(record,catalog,args.max_views)
        nd=distinct_positions(members,catalog)
        if nd<args.min_views:
            omitted.append({'query':record['query_name'],'reason':'insufficient_distinct_camera_positions',
                            'distinct_positions':nd,'source':str(path)})
            continue
        rid='region_'+digest({'catalog':catalog.id,'seed':catalog.rows[q]['observation_id']})[:12]
        if rid in seen_ids:raise ValueError('Duplicate seed region ID')
        seen_ids.add(rid)
        folder=out/'regions'/rid;folder.mkdir(parents=True,exist_ok=True)
        done=folder/'complete.json'
        # Verify original association outputs (not just their JSON names).
        for n,e in record['files'].items():checked(path.parent/n,e)
        if record['run_id']!=recipe['run_id']:raise ValueError('Wrong query run')
        if done.exists():
            st=read_json(done)
            if st['scene_id']!=config['scene_id']:raise ValueError('Region belongs to a different build')
            for n,e in st['files'].items():checked(folder/n,e)
            for i in members:store.frame(catalog.rows[i]['frame_id'])
            regions.append(read_json(folder/'region.json'))
            print(f'[{number}/{len(records)}] reused {rid}',flush=True);continue
        xyz=[];rgb=[];uv=[];oi=[];si=[];views=[];vectors=[];files=[]
        for local,i in enumerate(members):
            r=catalog.rows[i];surface=store.surface(i);frame=store.frame(r['frame_id'])
            if r['frame_status']!='keep' or r['mask_status']!='keep' or not r['eligible_geometry']:
                raise ValueError('Unexpected non-keep observation')
            xyz.append(surface.xyz);rgb.append(surface.rgb);uv.append(surface.uv)
            oi.append(np.full(len(surface.xyz),local,np.int32));si.append(np.full(len(surface.xyz),local,np.int32))
            masked,context,crop=crop_pair(frame['rgb'],surface.mask)
            sub=folder/'views';sub.mkdir(exist_ok=True)
            masked.save(sub/f'v{local:02d}_masked.png');context.save(sub/f'v{local:02d}_context.png')
            # Source overview is for human review only, not the model's crop input.
            from scipy.ndimage import binary_erosion
            overview=frame['rgb'].copy();edge=surface.mask & ~binary_erosion(surface.mask)
            overview[edge]=[255,200,20]
            im=Image.fromarray(overview);im.thumbnail((640,480),Image.Resampling.LANCZOS);im.save(sub/f'v{local:02d}_overview.jpg',quality=90)
            for suffix in ('masked.png','context.png','overview.jpg'):files.append(f'views/v{local:02d}_{suffix}')
            obs={'local_index':local,'catalog_index':i,'observation_id':r['observation_id'],
                 'frame_id':r['frame_id'],'mask_index':r['mask_index'],'station_id':r['station_id'],
                 'relation':'seed' if local==0 else 'directly_supported_by_seed',
                 'frame_status':'keep','mask_status':'keep','geometry_eligible':True,
                 'camera_to_map':surface.camera['camera_to_map'],'K':surface.camera['K'],
                 'camera_position_map_m':r['camera_position'],'crop_xyxy':crop,
                 'supported_pixels_total':surface.support_pixels,'retained_samples':len(surface.xyz),
                 'source_frame':str(Path(catalog.source['dataset'])/'frames'/r['frame_id']),
                 'mask_archive':str(Path(catalog.source['segmentation'])/'frames'/r['frame_id']/'masks.npz'),
                 'masked_crop':f'views/v{local:02d}_masked.png','context_crop':f'views/v{local:02d}_context.png',
                 'overview':f'views/v{local:02d}_overview.jpg'}
            views.append(obs);vectors.append(surface.crop)
        xyz=np.concatenate(xyz);rgb=np.concatenate(rgb);uv=np.concatenate(uv);oi=np.concatenate(oi);si=np.concatenate(si)
        fused=fuse_samples(xyz,rgb,oi,si,args.voxel)
        write_npz(folder/'samples.npz',xyz=xyz,rgb=rgb,source_uv=uv,observation_index=oi,station_index=si)
        write_npz(folder/'surface.npz',**fused)
        export_ply(folder/'surface.ply',fused['xyz'],fused['rgb'])
        consensus=fused['station_count']>=2
        export_ply(folder/'multiview_voxels.ply',fused['xyz'][consensus],fused['rgb'][consensus])
        vectors=np.asarray(vectors,np.float32)
        write_npz(folder/'descriptors.npz',per_observation_crop=vectors,aggregate_crop=unit(vectors.mean(axis=0)))
        bounds=bounding_box(fused['xyz']);robust=np.percentile(fused['xyz'],[2,98],axis=0)
        obj={'region_id':rid,'status':'tentative_multiview_region','semantic_label':None,
             'confirmed_instance_id':None,'navigation_goal':None,'source_query':record['query_name'],
             'source_hypothesis_id':record['hypothesis']['hypothesis_id'],'seed_observation_id':catalog.rows[q]['observation_id'],
             'source_query_path':str(path),'member_count':len(views),'distinct_camera_positions_10cm':nd,
             'views':views,'point_count':len(fused['xyz']),'raw_sample_count':len(xyz),
             'multiview_voxel_count':int(consensus.sum()),'voxel_m':args.voxel,'bbox':bounds,
             'surface_median_map_m':np.median(fused['xyz'],axis=0).tolist(),
             'robust_surface_bounds_p02_p98_m':robust.tolist(),'color':color_for(rid),
             'structural_hints':sorted(set(h for i in members for h in catalog.rows[i].get('hints',[]))),
             'partial_overlap_observation_ids':[e['candidate_observation_id'] for e in record['edges'] if e['status']=='partial_overlap'],
             'all_direct_supported_observation_ids':[e['candidate_observation_id'] for e in record['edges'] if e['status']=='supported'],
             'notes':['Group is seed-supported, not all-pairs confirmed or transitively merged.',
                      'Samples come from rendered expected depth, not original Gaussian indices.',
                      'Voxel multi-view count is agreement at this discretization, not calibrated confidence.',
                      'Groups can overlap; source masks and point assignments are not exclusive.',
                      'Bounding box encloses observed samples only; not a complete object collision envelope.']}
        write_json(folder/'region.json',obj)
        files+=['samples.npz','surface.npz','surface.ply','multiview_voxels.ply','descriptors.npz','region.json']
        write_json(done,{'scene_id':config['scene_id'],'files':file_records(folder,files)})
        regions.append(obj);print(f'[{number}/{len(records)}] {rid}: {len(views)} views | {len(fused["xyz"]):,} voxels | {consensus.sum():,} multiview',flush=True)
    if not regions:raise ValueError('No groups meet minimum view support. Inspect Stage 4A before reducing it.')
    view_records=[{k:v for k,v in r.items() if k not in ('views',)} for r in regions]
    plan=read_json(Path(catalog.source['dataset'])/'plan.json')
    scene={'schema':'valmet.semantic_scene.v1','scene_id':config['scene_id'],'association_run':str(run),
           'capture_plan_hash':catalog.source['capture_plan_hash'],'coordinate_frame':'map',
           'scale_m_per_scene_unit':float(plan['config']['scale']),
           'floor_y_gaussian':float(plan['config'].get('floor_y',1.1)),
           'geometry_type':'nonexclusive_expected_depth_visible_surface_hypotheses',
           'query_count':len(records),'region_count':len(regions),'regions':view_records,
           'omitted_queries':omitted,'notes':['No automatic semantic labels until suggest-labels is run.',
               'Labels from manual review and model suggestions are stored separately.',
               'Unobserved/unassigned environment stays unlabelled. No complete-scene segmentation claim.']}
    write_json(out/'scene.json',scene)
    summary={'scene_id':config['scene_id'],'input_queries':len(records),'exported_regions':len(regions),
             'omitted_insufficient_views':len(omitted),'surface_points_sum_with_overlap':sum(r['point_count'] for r in regions),
             'semantic_labels_confirmed':0,'seconds':time.perf_counter()-started,
             'provisional_scale':scene['scale_m_per_scene_unit']}
    write_json(out/'summary.json',summary)
    with (out/'regions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['region_id','views','points','multiview_voxels','x','y','z','dx','dy','dz'])
        for r in regions:w.writerow([r['region_id'],r['member_count'],r['point_count'],r['multiview_voxel_count'],*r['bbox']['center'],*r['bbox']['extent']])
    print(json.dumps(summary,indent=2));return scene
