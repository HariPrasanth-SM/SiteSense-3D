from __future__ import annotations
import math, time
from pathlib import Path
import numpy as np
from avatar_demo.core import FloorMap,Planner,PlanningError
from .common import read_json,atomic_json,sha
from .workbook import load,asset_database
from .regions import Regions

def prepare(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    wb=load(args.workbook);db=asset_database(wb)
    fmap=FloorMap.load(args.map);planner=Planner(fmap,0,0)
    regions=Regions(args.regions,args.enhancement)
    scale=float(regions.manifest.get('scale_m_per_scene_unit',1.43));floor=float(regions.manifest.get('floor_y_gaussian',1.1))
    config={'schema':'valmet.service_demo.v1','map':str(Path(args.map).resolve()),'map_sha256':sha(args.map),
            'ply':str(Path(args.ply).resolve()),'regions':str(regions.root),'enhancement':str(regions.enh) if regions.enh else None,
            'association':str(Path(args.association).resolve()),'repo':str(Path(args.repo).resolve()),
            'weights':str(Path(args.weights).resolve()),'scale':scale,'floor_y':floor,
            'source_workbook_sha256':wb['source_sha256'],'default_camera_height':1.7,
            'default_speed':0.25,'default_yaw_degrees':90,'default_radius':0.0,'default_margin':0.0,
            'min_similarity':0.70,'search_count':200,'nearby_radius':2.0,
            'warning':'Synthetic workbook, explicitly demo-bound asset IDs, provisional 1.43 scene scale, static 2D kinematic motion; no physical robot control.'}
    if (out/'config.json').exists() and read_json(out/'config.json')!=config:
        raise ValueError('Output was prepared with different inputs/settings. Use a NEW --out directory.')
    atomic_json(out/'config.json',config);atomic_json(out/'workbook.json',db)
    bindings={};start=planner.auto_start()
    candidates=[]
    for rid,r in regions.items.items():
        if r['review'].get('decision')=='rejected':continue
        target=np.asarray(r['bbox']['center'],float)
        try:p=planner.plan(start,target[:2],standoff=.8,max_approach=3.5)
        except PlanningError:continue
        size=np.linalg.norm(r['bbox']['extent']);rank=0 if r['review'].get('kind') in ('equipment','part') else 1
        if r['review'].get('kind')=='structure':rank=3
        candidates.append((rank,size,rid,target,p.endpoint))
    selected=[]
    while candidates and len(selected)<len(db['assets']):
        if not selected:best=min(range(len(candidates)),key=lambda i:(candidates[i][0],abs(candidates[i][1]-.8),candidates[i][2]))
        else:
            def merit(c):
                separation=min(np.linalg.norm(c[3][:2]-s[3][:2]) for s in selected)
                return min(separation,4)-.45*c[0]-.08*max(0,c[1]-3)
            best=max(range(len(candidates)),key=lambda i:(merit(candidates[i]),candidates[i][2]))
        selected.append(candidates.pop(best))
    for i,(aid,a) in enumerate(sorted(db['assets'].items())):
        b={'asset_id':aid,'status':'unmapped','region_id':None,'map_xyz':None,'anchor_original':a['anchor_original'],
           'note':'Original coordinates are synthetic placeholders; not registered to this scan.'}
        if args.demo_bindings and i<len(selected):
            _,_,rid,target,approach=selected[i]
            b.update(status='procedural_demo_binding',region_id=rid,map_xyz=target.tolist(),
                     suggested_approach_xy=approach.tolist(),
                     note='Explicit procedural demo link. Workbook class/identity is NOT inferred from the geometry or validated against this physical object.')
        bindings[aid]=b
    # Never overwrite user-reviewed bindings on repeated prepare.
    if not (out/'asset_bindings.json').exists():atomic_json(out/'asset_bindings.json',{'revision':0,'assets':bindings})
    else:bindings=read_json(out/'asset_bindings.json')['assets']
    stats={'assets':len(db['assets']),'mapped_for_demo':sum(bool(b['map_xyz']) for b in bindings.values()),
           'regions_available':len(regions.items),'source_counts':db['counts'],'provisional_start_xy':start.tolist(),
           'coordinates_are_placeholders':True,'asset_mapping_policy':'Explicit demo binding, never silently trusted workbook coordinates',
           'map_state':'candidate existing map; unknown blocked','time':time.time()}
    atomic_json(out/'prepare_summary.json',stats)
    print('PREPARED',out,flush=True)
    print(f"Assets: {stats['assets']} | demo-bound: {stats['mapped_for_demo']} | source regions: {stats['regions_available']}",flush=True)
    print('Workbook anchors are PLACEHOLDERS. The dashboard visibly labels demo bindings.\nNo original source files were changed.',flush=True)
