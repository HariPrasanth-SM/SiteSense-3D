"""Two-row layout and continuous camera/object transforms; no CUDA dependency."""
from __future__ import annotations
import csv
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from semantic_animation.plan import font, image_fit
from .common import *


def fit_distance(points_camera, fov, aspect, margin_x=.82, margin_y=.70):
    tan=math.tan(math.radians(fov)/2)
    a=np.asarray(points_camera)
    return max(.6,float(np.max(np.maximum(np.abs(a[:,0])/(tan*aspect*margin_x),
        np.abs(a[:,1])/(tan*margin_y))-a[:,2]))+.12)


def create_motion(p, end_camera, end_translation, selected_ids, *, fps, tail_frames,
                  formation, pullback, rotation_deg, gap, forward_distance, hero_scale=1.):
    """Global original region order retained. Only selected indices change pose/scale."""
    end_camera=valid_camera(end_camera);G=end_camera[:3,:3];right=G[:,0];up=-G[:,1];forward=G[:,2]
    rows=p['regions'];N=len(rows);lookup={r['region_id']:i for i,r in enumerate(rows)}
    selected=[lookup[r] for r in selected_ids];M=len(selected);cols=math.ceil(M/2)
    centers=np.asarray([r['center'] for r in rows],float)
    start_centers=centers+end_translation
    anchor=np.median(start_centers,axis=0)
    distance0=abs(float((anchor-end_camera[:3,3])@forward))
    shift=float(forward_distance) if forward_distance is not None else max(1.,distance0*.12)
    gallery_center=anchor-forward*shift
    total_last=(tail_frames-1)/fps
    if total_last <= formation+pullback+.3:
        raise ValueError('Finale is too short: increase total duration or shorten formation/pullback (need some pan time).')
    # Rotational envelope per region, using original declared display bounds, never shrinking geometry.
    bounds={}; signs={j:(1 if k%2==0 else -1) for k,j in enumerate(selected)}
    for j in selected:
        r=rows[j];c=centers[j];points=corners(r['display_bounds_min'],r['display_bounds_max'])-c
        samples=[]
        for angle in np.linspace(0,math.radians(rotation_deg)*signs[j],17):
            samples.append((points@rz(angle).T)*hero_scale@G)
        a=np.concatenate(samples)
        half=np.maximum(np.max(np.abs(a),axis=0)*1.04,.025)
        bounds[j]=half
    colhalf=np.array([max(bounds[j][0] for k,j in enumerate(selected) if k//2==col) for col in range(cols)])
    rowhalf=np.array([max((bounds[j][1] for k,j in enumerate(selected) if k%2==row),default=.1) for row in range(2)])
    x=np.zeros(cols)
    for col in range(1,cols):x[col]=x[col-1]+colhalf[col-1]+colhalf[col]+gap
    x-=((x[0]-colhalf[0])+(x[-1]+colhalf[-1]))/2
    rowy=np.array([rowhalf[1]+gap/2,-rowhalf[0]-gap/2]) if M>1 else np.zeros(2)
    target=start_centers.copy();local_positions={};envelope=[];owners=[]
    for k,j in enumerate(selected):
        col,row=divmod(k,2);local=np.array([x[col],-rowy[row],0.])
        local_positions[j]=local;target[j]=gallery_center+G@local
        pts=corners(-bounds[j],bounds[j])+local;envelope.extend(pts);owners.extend([col]*8)
    envelope=np.array(envelope);owners=np.array(owners)
    # First four = first two columns with two rows. Camera slides continuously across columns.
    windows=[]
    for col in range(max(1,cols-1)):
        ps=envelope[(owners>=col)&(owners<=min(cols-1,col+1))]
        cx=(ps[:,0].min()+ps[:,0].max())/2
        cy=(ps[:,1].min()+ps[:,1].max())/2
        ref=np.array([cx,cy,0.]);d=fit_distance(ps-ref,p['fov_y_deg'],p['width']/p['height'])
        windows.append((ref,d))
    close_dist=max(d for _,d in windows)
    left_ref=windows[0][0];right_ref=windows[-1][0]
    full_ref=np.array([(envelope[:,0].min()+envelope[:,0].max())/2,
                       (envelope[:,1].min()+envelope[:,1].max())/2,0.])
    full_dist=fit_distance(envelope-full_ref,p['fov_y_deg'],p['width']/p['height'],.84,.72)
    # Never zoom IN during the requested final pullback.
    full_dist=max(full_dist,close_dist*1.12)
    def cam(ref,dist):
        T=np.eye(4);T[:3,:3]=G;T[:3,3]=gallery_center+G@ref-forward*dist;return T
    left_cam=cam(left_ref,close_dist);right_cam=cam(right_ref,close_dist);final_cam=cam(full_ref,full_dist)
    F=tail_frames;t=np.arange(F)/fps
    camposes=np.repeat(end_camera[None],F,axis=0)
    cc=np.repeat(start_centers[None],F,axis=0)
    angle=np.zeros((F,N));opacity=np.ones((F,N));sizes=np.ones((F,N))
    stage=[]
    pan_end=total_last-pullback
    for i,sec in enumerate(t):
        formation_weight=float(eased(sec/formation))
        cc[i,selected]=start_centers[selected]*(1-formation_weight)+target[selected]*formation_weight
        sizes[i,selected]=1+(hero_scale-1)*formation_weight
        nonselected=np.ones(N,bool);nonselected[selected]=False
        opacity[i,nonselected]=1-float(eased(sec/min(.9,formation*.8)))
        # Gentle rotation begins after initial formation; alternating direction, no camera-facing billboards.
        rot_weight=float(eased((sec-formation*.35)/(total_last-formation*.35)))
        for j in selected:angle[i,j]=math.radians(rotation_deg)*signs[j]*rot_weight
        if sec<formation:
            camposes[i,:3,3]=end_camera[:3,3]*(1-formation_weight)+left_cam[:3,3]*formation_weight
            stage.append('Selected regions move forward')
        elif sec<pan_end:
            w=float(eased((sec-formation)/(pan_end-formation)))
            camposes[i,:3,3]=left_cam[:3,3]*(1-w)+right_cam[:3,3]*w
            stage.append('Two-row close-up | left to right')
        else:
            w=float(eased((sec-pan_end)/pullback))
            camposes[i,:3,3]=right_cam[:3,3]*(1-w)+final_cam[:3,3]*w
            stage.append('Selected collection | final pullback')
    transforms=np.tile(np.eye(4),(F,M,1,1))
    for k,j in enumerate(selected):
        for i in range(F):
            R=rz(angle[i,j])*sizes[i,j];transforms[i,k,:3,:3]=R
            transforms[i,k,:3,3]=cc[i,j]-R@centers[j]
    layout={'columns':cols,'rows':2,'order':'column-major: 1 top-left, 2 bottom-left, 3 top-second, 4 bottom-second',
            'selected_indices':selected,'hero_center_map':gallery_center.tolist(),'forward_shift_map_units':shift,
            'gap_map_units':gap,'hero_uniform_scale':hero_scale,'close_camera_distance':close_dist,
            'final_camera_distance':full_dist,'first_four_ids':selected_ids[:4],
            'positions':[{'region_id':rows[j]['region_id'],'display_label':rows[j]['display_label'],
                 'order':k+1,'row':k%2,'column':k//2,'center_map':target[j].tolist(),
                 'local_center_camera_axes':local_positions[j].tolist(),
                 'rotation_end_deg':rotation_deg*signs[j],'rotation_envelope_half':bounds[j].tolist()} for k,j in enumerate(selected)],
            'timing_seconds':{'formation':formation,'pan':pan_end-formation,'pullback':pullback},
            'notes':['Pure presentation transforms; not physical motion or true object rearrangement.',
                     'Bounds include the source Gaussian display margin, not infinite Gaussian tails.',
                     'Object scale defaults to 1; no per-object size normalization.']}
    return {'time_s':t.astype(np.float64),'camera_to_map':camposes.astype(np.float64),
            'center_map':cc.astype(np.float64),'yaw_rad':angle.astype(np.float64),
            'opacity_weight':opacity.astype(np.float32),'uniform_scale':sizes.astype(np.float32),
            'selected_transform_map':transforms.astype(np.float64),'stage':np.asarray(stage)},layout


def prepare(args):
    base,p=base_plan(args.base);selection,ids=selection_ids(args.selection,p)
    out=args.out.expanduser().resolve()
    if out==base or out.is_relative_to(base):raise ValueError('Use a NEW output folder outside the successful base animation.')
    if out.exists() and any(out.iterdir()):raise ValueError('Output is not empty. Choose a new folder; render resumes existing plans.')
    fps=p['fps'];total=round(args.duration*fps)
    intro=round((args.intro_duration if args.intro_duration is not None else p['frame_count']/fps)*fps)
    tail=total-intro
    if intro<2 or tail<2:raise ValueError('Duration must exceed the intro duration and allow a finale.')
    # Bind all source assets and final transforms. Original movie's frames are validated when imported.
    hashes=file_input_hashes(base,p)
    with np.load(base/'motion.npz',allow_pickle=False) as a:
        c=a['camera_to_map'][-1].astype(float);delta=a['translation_map'][-1].astype(float)
    motion,layout=create_motion(p,c,delta,ids,fps=fps,tail_frames=tail,formation=args.formation,
        pullback=args.pullback,rotation_deg=args.rotation_deg,gap=args.gap,forward_distance=args.forward_distance,
        hero_scale=args.hero_scale)
    out.mkdir(parents=True)
    write_json(out/'source_plan.json',p);write_json(out/'selection.json',selection)
    write_json(out/'layout.json',layout);save_npz(out/'motion.npz',**motion)
    # Reuse exactly the original rendered prefix, or explicitly retime with nearest-source-frame selection.
    intro_indices=np.rint(np.linspace(0,p['frame_count']-1,intro)).astype(np.int64)
    np.save(out/'intro_source_indices.npy',intro_indices)
    with (out/'camera_path.jsonl').open('w') as f:
        for i in range(tail):
            f.write(__import__('json').dumps({'tail_frame':i,'video_frame':intro+i,'time_s':(intro+i)/fps,
                'camera_to_map':motion['camera_to_map'][i].tolist(),
                'camera_to_world':camera_map_to_scene(motion['camera_to_map'][i],p['scale'],p['floor_y']).tolist(),
                'K':p['K'],'selected_ids':ids,'selected_transform_map':motion['selected_transform_map'][i].tolist(),
                'object_center_map':motion['center_map'][i].tolist(),'object_yaw_rad':motion['yaw_rad'][i].tolist(),
                'opacity_weight':motion['opacity_weight'][i].tolist(),'uniform_scale':motion['uniform_scale'][i].tolist()})+'\n')
    code_hashes={str(x.relative_to(Path(__file__).parent)):sha(x) for x in Path(__file__).parent.rglob('*.py')}
    plan={'schema':'valmet.semantic_showcase.v1','base':str(base),'base_plan_id':p['plan_id'],
        'fps':fps,'frame_count':total,'intro_frames':intro,'tail_frames':tail,'duration_s':total/fps,
        'base_duration_s':p['frame_count']/fps,'intro_duration_s':intro/fps,
        'width':p['width'],'height':p['height'],'K':p['K'],'background':p['background'],
        'selected_ids':ids,'all_region_count':p['region_count'],'selection_count':len(ids),
        'formation_s':args.formation,'pullback_s':args.pullback,'rotation_deg':args.rotation_deg,
        'gaussian_hashes':hashes,'base_motion_sha256':sha(base/'motion.npz'),
        'work_files':{name:sha(out/name) for name in ['source_plan.json','selection.json','layout.json','motion.npz','camera_path.jsonl','intro_source_indices.npy']},
        'notes':['Intro copied from successful animation frames; all original objects remain there.',
                 'Selected 3D Gaussian means and quaternion covariance axes rotate together.',
                 'SH appearance evaluated using inverse-rotated object-local viewing directions.',
                 'No training, scene editing, asset label changes, or CUDA package upgrade.',
                 'Base asset directory remains a dependency. Keep it.']}
    plan['plan_id']=digest(plan);write_json(out/'plan.json',plan)
    # Saved gallery map is a schematic of centers and extents, not a GPU rendering.
    make_layout_preview(out,p,layout)
    print(f'Prepared {total/fps:.2f}s: {intro/fps:.2f}s original reveal + {tail/fps:.2f}s showcase.',flush=True)
    print(f'{len(ids)} selected / {p["region_count"]} base regions. Order is paired columns; first four are leftmost.',flush=True)
    print('Layout:',out/'showcase_layout.jpg',flush=True)
    for r in layout['positions']:
        if max(r['rotation_envelope_half'])>3:
            print(f'NOTE: {r["region_id"]} has a large display bound; it may make nearby objects look small.',flush=True)
    print('No original movie/asset was changed. Next: render --out "'+str(out)+'" --preview',flush=True)


def make_layout_preview(out,p,layout):
    # Contact-sheet order exactly matches two display rows; uses source masked crops, not fabricated geometry.
    base=Path(read_json(out/'plan.json')['base']);cols=layout['columns'];W=240;H=240
    sheet=Image.new('RGB',(max(2,cols)*W,2*H+50),(16,23,32));d=ImageDraw.Draw(sheet)
    d.text((12,8),'Selected showcase order — top/bottom pairs, camera pans left to right',font=font(17),fill='white')
    lookup={r['region_id']:r for r in p['regions']}
    for item in layout['positions']:
        r=lookup[item['region_id']];x=item['column']*W;y=50+item['row']*H
        evidence=r.get('evidence',[])
        if evidence and evidence[0].get('masked_crop'):
            src=checked_evidence(base,r,evidence[0]['masked_crop']);sheet.paste(image_fit(src,(W-12,H-65)),(x+6,y+4))
        d.text((x+6,y+H-57),f'{item["order"]:02d} | '+r.get('display_label','')[:24],font=font(14),fill='white')
        d.text((x+6,y+H-36),r['region_id'][:31],font=font(11),fill=(180,195,210))
        d.text((x+6,y+H-20),f'Row {item["row"]+1}, column {item["column"]+1}',font=font(11),fill=(150,180,200))
    sheet.save(out/'showcase_layout.jpg',quality=92)
