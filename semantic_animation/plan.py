"""Package existing evidence/geometry and generate an explicit cinematic timeline."""
from __future__ import annotations
import csv
import html
import math
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image,ImageDraw,ImageFont
from .common import *

STRUCTURES={'wall','floor','ceiling','room','wall section','floor surface','ceiling surface'}


def font(size):
    for f in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(f).exists(): return ImageFont.truetype(f,size)
    return ImageFont.load_default()


def image_fit(path, wh):
    with Image.open(path) as im:
        im=im.convert('RGB');im.thumbnail(wh)
        out=Image.new('RGB',wh,(25,30,40));out.paste(im,((wh[0]-im.width)//2,(wh[1]-im.height)//2));return out


def catalog(enh, *, allow_small=False, include_rejected=False):
    enh=Path(enh).expanduser().resolve();cfg=read_json(enh/'config.json');src=Path(cfg['source_scene'])
    scene=read_json(src/'scene.json')
    if scene['scene_id']!=cfg['source_scene_id']: raise ValueError('Source scene identity has changed.')
    rev=read_json(src/'review.json').get('regions',{}) if (src/'review.json').exists() else {}
    sug=read_json(src/'semantic_suggestions.json').get('regions',{}) if (src/'semantic_suggestions.json').exists() else {}
    out=[]
    for r in scene['regions']:
        rid=r['region_id'];folder=enh/'regions'/rid
        if not (folder/'complete.json').exists() or not (folder/'gaussians_original.ply').exists():continue
        e=read_json(folder/'enhancement.json');n=e.get('selected_gaussians',0)
        if n<(1 if allow_small else 20):continue
        rr=read_json(src/'regions'/rid/'region.json');rv=rev.get(rid,{})
        if rv.get('decision')=='rejected' and not include_rejected:continue
        box=e['clean_bbox'];lo=np.array(box['min']);hi=np.array(box['max']);extent=hi-lo
        label=rv.get('semantic_label') or ''
        if not label and sug.get(rid,{}).get('suggested_label'): label=sug[rid]['suggested_label']+' ?'
        structural=rv.get('kind')=='structure' or label.strip(' ?').lower() in STRUCTURES
        score=(1000 if rv.get('decision')=='accepted' else 0)+(80 if rv.get('kind') in ['equipment','part'] else 0)
        score+=5*min(rr.get('member_count',1),9)+np.log1p(n)
        row={'region_id':rid,'label':label,'review':rv,'structural':structural,'score':float(score),
             'point_count':int(e['cleaning']['kept']),'gaussians':int(n),'extent':extent.tolist(),
             'min':lo.tolist(),'max':hi.tolist(),'center':((lo+hi)/2).tolist(),'color':rr.get('color',[70,190,230]),
             'views':rr['views'],'source_region':str(src/'regions'/rid),'enhanced_region':str(folder),
             'source_hypothesis_id':rr.get('source_hypothesis_id')}
        out.append(row)
    return cfg,src,sorted(out,key=lambda r:(-r['score'],r['region_id']))


def choose(rows,args):
    if getattr(args,'all_regions',False):
        if not rows:raise ValueError('No prepared nonempty Gaussian regions. Inspect the enhancement output first.')
        return [dict(r) for r in rows], [
            {'region_id':r['region_id'],'selected':True,
             'reason':'explicit --all-regions: no count/structure/extent/overlap filtering'} for r in rows]
    explicit=[r.strip() for r in args.regions.split(',') if r.strip()] if args.regions else None
    log=[];selected=[];indices={}
    if explicit:
        table={r['region_id']:r for r in rows}
        missing=[r for r in explicit if r not in table]
        if missing:raise ValueError(f'Regions missing a prepared Gaussian subset, rejected, or invalid: {missing}')
        pool=[table[r] for r in dict.fromkeys(explicit)]
    else: pool=rows
    for r in pool:
        reason=None
        if not explicit:
            if r['structural']:reason='labelled structure; omitted from automatic showreel selection'
            elif max(r['extent'])>args.max_extent:reason='larger than automatic presentation extent limit'
            elif max(r['extent'])<.08:reason='too small for automatic wide-shot selection'
        p=Path(r['enhanced_region'])/'gaussian_indices.npy'
        ids=np.unique(np.load(p,allow_pickle=False)) if p.exists() else np.zeros(0,np.int64)
        if reason is None and not explicit:
            for old in selected:
                other=indices[old['region_id']]
                if min(len(ids),len(other)) and len(np.intersect1d(ids,other,assume_unique=True))/min(len(ids),len(other))>.65:
                    reason=f'large shared-Gaussian overlap with {old["region_id"]}; showreel-only duplicate suppression';break
        if reason is None and not explicit and len(selected)>=args.objects:reason='automatic presentation budget'
        if reason: log.append({'region_id':r['region_id'],'selected':False,'reason':reason});continue
        selected.append(dict(r));indices[r['region_id']]=ids
        log.append({'region_id':r['region_id'],'selected':True,'reason':'explicit selection' if explicit else 'presentation heuristic, inspect preview'})
    if not selected:raise ValueError('No eligible regions. Inspect enhanced Gaussian counts; use --regions for explicit selection or increase --max-extent.')
    return selected,log


def choose_camera(rows,scale,floor,args):
    if args.camera:
        return load_camera(args.camera,scale,floor),{'method':'user_camera','file':str(args.camera.resolve())}
    centers=np.asarray([r['center'] for r in rows]);best=None;best_score=-1e20
    tan=math.tan(math.radians(args.fov)/2);aspect=args.width/args.height
    seen=set()
    for r in rows:
        for v in r['views']:
            key=v['frame_id']
            if key in seen:continue
            seen.add(key);T=valid_camera(v['camera_to_map']);p=(centers-T[:3,3])@T[:3,:3]
            visible=(p[:,2]>.25)&(p[:,2]<18)&(np.abs(p[:,0])<.85*tan*aspect*p[:,2])&(np.abs(p[:,1])<.85*tan*p[:,2])
            score=visible.sum()*10-np.linalg.norm(T[:3,3]-np.median(centers,axis=0))*.1
            if score>best_score:best_score=score;best=T;source={'method':'support-camera heuristic; not collision/appearance verified','frame_id':key,'source_frame':v['source_frame']}
    if best is None:raise ValueError('No valid source camera. Supply --camera.')
    return best,source


def prepare(args):
    from plyfile import PlyData
    cfg,src,rows=catalog(args.enhancement,allow_small=getattr(args,'all_regions',False),
                         include_rejected=getattr(args,'include_rejected',False))
    out=args.out.expanduser().resolve()
    if (out/'plan.json').exists():raise ValueError('Plan already exists; use render to resume or choose a new --out.')
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty/new output directory for prepare.')
    if args.duration<8 or args.duration>60 or not 1<=args.fps<=60:raise ValueError('Use duration 8–60 s and fps 1–60.')
    if args.width<320 or args.height<240 or args.width%2 or args.height%2:raise ValueError('Use even dimensions at least 320x240.')
    selected,log=choose(rows,args);out.mkdir(parents=True,exist_ok=True);assets=out/'assets';assets.mkdir()
    scale=float(cfg['scale']);floor=float(cfg['floor_y']);camera,cam_source=choose_camera(selected,scale,floor,args)
    ply=Path(args.scene_ply or cfg.get('gaussian_ply','')).expanduser().resolve()
    if not ply.is_file():raise ValueError('Original room Gaussian PLY missing; supply --scene-ply.')
    print('Hashing source room PLY and copying selected assets...',flush=True)
    source_hash=sha(ply)
    if cfg.get('gaussian_sha256') and source_hash!=cfg['gaussian_sha256']:raise ValueError('Room PLY differs from the source used to extract these subsets.')
    # A portable input bundle includes room PLY only when explicitly requested.
    if args.portable:
        shutil.copy2(ply,assets/'environment_original.ply');room_ref='assets/environment_original.ply'
    else:room_ref=str(ply)
    if args.environment:
        envpath=args.environment.expanduser().resolve();xyz,rgb=read_cloud(envpath)
        envsource={'path':str(envpath),'sha256':sha(envpath),'frame':'already in map; no second transform'}
    else:
        v=PlyData.read(str(ply),mmap=True)['vertex'].data
        rng=np.random.default_rng(args.seed);ids=np.sort(rng.choice(len(v),min(len(v),args.env_points),replace=False))
        xyz=scene_to_map(np.column_stack([v[k][ids] for k in ['x','y','z']]),scale,floor).astype(np.float32)
        dc=np.column_stack([v[f'f_dc_{i}'][ids] for i in range(3)])
        rgb=np.rint(np.clip(.5+.28209479177387814*dc,0,1)*255).astype(np.uint8)
        envsource={'path':str(ply),'derivation':'original Gaussian centers + SH DC; point display only'}
    xyz,rgb=sample_cloud(xyz,rgb,args.env_points,args.seed)
    good=np.isfinite(xyz).all(1);xyz,rgb=xyz[good],rgb[good]
    if not len(xyz):raise ValueError('Environment point cloud is empty.')
    save_npz(assets/'environment_points_map.npz',xyz=xyz,rgb=rgb)
    write_ply(assets/'environment_points_map.ply',xyz,rgb)
    for i,r in enumerate(selected):
        rid=r['region_id'];dest=assets/rid;dest.mkdir();enh=Path(r['enhanced_region']);raw=Path(r['source_region'])
        r['display_id']=f'R{i+1:02d}';r['display_label']=r['label'] or r['display_id'];r['asset_dir']=f'assets/{rid}'
        for name in ['surface_clean.ply','surface_clean.npz','gaussians_original.ply','gaussian_indices.npy','enhancement.json']:
            if (enh/name).exists():shutil.copy2(enh/name,dest/name)
        shutil.copy2(raw/'region.json',dest/'region.json')
        px,pc=read_cloud(dest/'surface_clean.npz');px,pc=sample_cloud(px,pc,args.object_points,args.seed+i)
        save_npz(dest/'display_points_map.npz',xyz=px,rgb=pc)
        g=PlyData.read(str(dest/'gaussians_original.ply'),mmap=True)['vertex'].data
        gm=scene_to_map(np.column_stack([g[k] for k in ['x','y','z']]),scale,floor)
        sigma=float(np.max(np.exp(np.clip(np.column_stack([g[f'scale_{j}'] for j in range(3)]),-30,20))))*scale
        # 2-sigma display margin, not a claim that infinite Gaussian tails are fully bounded.
        lo=np.minimum(np.array(r['min']),gm.min(0)-2*sigma);hi=np.maximum(np.array(r['max']),gm.max(0)+2*sigma)
        r['display_bounds_min']=lo.tolist();r['display_bounds_max']=hi.tolist();r['max_gaussian_sigma_m']=sigma
        (dest/'evidence').mkdir()
        evidence=[]
        for j,v in enumerate(r['views'][:3]):
            ev={'frame_id':v['frame_id'],'mask_index':v['mask_index'],'station_id':v['station_id'],
                'observation_id':v.get('observation_id'),'camera_to_map':v['camera_to_map'],'K':v['K']}
            for key,suffix,base in [('masked_crop','masked.png',raw),('context_crop','context.png',raw),('overview','overview.jpg',raw)]:
                p=base/v.get(key,'')
                if p.is_file():
                    name=f'evidence/v{j:02d}_{suffix}';shutil.copy2(p,dest/name);ev[key]=name
            overlay=enh/'views'/f'v{j:02d}_overlay.jpg'
            if overlay.exists():name=f'evidence/v{j:02d}_overlay.jpg';shutil.copy2(overlay,dest/name);ev['overlay']=name
            evidence.append(ev)
        write_json(dest/'evidence.json',evidence);r['evidence']=evidence
        r['source_views_count']=len(r['views']);r.pop('views',None)
    # Gallery plane has camera X/right and camera-up axes. Object rotations remain unchanged.
    f=camera[:3,2].copy();yaw=math.atan2(f[1],f[0]);pitch=math.radians(-12.)
    forward=np.array([math.cos(yaw)*math.cos(pitch),math.sin(yaw)*math.cos(pitch),math.sin(pitch)])
    G=look_at([0,0,0],forward)[:3,:3];right=G[:,0];up=-G[:,1]
    cols=args.columns or math.ceil(math.sqrt(len(selected)*1.3));cols=min(cols,len(selected));nrows=math.ceil(len(selected)/cols)
    width=[];height=[]
    for r in selected:
        p=(corners(r['display_bounds_min'],r['display_bounds_max'])-r['center'])@G
        width.append(2*max(abs(p[:,0].min()),abs(p[:,0].max())))
        height.append(2*max(abs(p[:,1].min()),abs(p[:,1].max())))
    cw=max(width)+args.gap;ch=max(height)+args.gap
    gallery_center=np.median(np.asarray([r['center'] for r in selected]),axis=0)
    allcorners=[]
    for i,r in enumerate(selected):
        row,col=divmod(i,cols);items=min(cols,len(selected)-row*cols)
        center=gallery_center+right*((col-(items-1)/2)*cw)+up*(((nrows-1)/2-row)*ch)
        offset=center-np.array(r['center']);r['gallery_center_map']=center.tolist();r['translation_map']=offset.tolist()
        r['translation_gaussian']=(offset@ROT/scale).tolist()
        T=np.eye(4);T[:3,3]=offset;r['gallery_transform_map']=T.tolist()
        d=assets/r['region_id'];p,c=read_cloud(d/'surface_clean.npz');write_ply(d/'gallery_surface_map.ply',p+offset,c)
        gply=PlyData.read(str(d/'gaussians_original.ply'),mmap=False)
        g=gply['vertex'].data
        for j,key in enumerate(['x','y','z']):g[key]+=r['translation_gaussian'][j]
        gply.write(str(d/'gallery_gaussians_original_axes.ply'))
        write_json(d/'asset.json',r)
        allcorners.append(corners(r['display_bounds_min'],r['display_bounds_max'])+offset)
    p=(np.concatenate(allcorners)-gallery_center)@G;tan=math.tan(math.radians(args.fov)/2);aspect=args.width/args.height
    dist=max(3.,float(np.max(np.maximum(np.abs(p[:,0])/(tan*aspect*.80),np.abs(p[:,1])/(tan*.76))-p[:,2]))+1.)
    gallery_cam=np.eye(4);gallery_cam[:3,:3]=G;gallery_cam[:3,3]=gallery_center-forward*dist
    F=round(args.duration*args.fps);t=np.arange(F,dtype=float)/args.fps;u=np.linspace(0.,1.,F)
    slide=smooth(.30,.70,u);cams=np.zeros((F,4,4));q=smooth(.24,.72,u)
    rotations=Slerp([0,1],Rotation.from_matrix(np.stack([camera[:3,:3],G])))(q).as_matrix()
    cams[:]=np.eye(4);cams[:,:3,:3]=rotations;cams[:,:3,3]=camera[:3,3]*(1-q[:,None])+gallery_cam[:3,3]*q[:,None]
    offsets=np.zeros((F,len(selected),3));cloud=np.zeros((F,len(selected)));splat=np.zeros_like(cloud)
    for i,r in enumerate(selected):
        delay=.03*i/max(1,len(selected)-1)
        offsets[:,i,:]=smooth(.31+delay,.70+delay,u)[:,None]*r['translation_map']
        reveal=smooth(.73+.11*i/max(1,len(selected)-1),.82+.11*i/max(1,len(selected)-1),u)
        cloud[:,i]=smooth(.25,.36,u)*(1-reveal);splat[:,i]=reveal
    env_weight=1-smooth(.31,.54,u);room_weight=1-smooth(.14,.27,u)
    motion={'time_s':t.astype(np.float32),'camera_to_map':cams.astype(np.float32),
            'translation_map':offsets.astype(np.float32),'point_weight':cloud.astype(np.float32),
            'splat_weight':splat.astype(np.float32),'environment_point_weight':env_weight.astype(np.float32),
            'room_splat_weight':room_weight.astype(np.float32)}
    save_npz(out/'motion.npz',**motion)
    K=np.array([[args.height/(2*tan),0,args.width/2],[0,args.height/(2*tan),args.height/2],[0,0,1]])
    plan={'schema':'valmet.semantic_animation.v1','duration_s':args.duration,'fps':args.fps,'frame_count':F,
          'width':args.width,'height':args.height,'fov_y_deg':args.fov,'K':K.tolist(),'scale':scale,'floor_y':floor,
          'room_ply':room_ref,'room_sha256':source_hash,'environment_source':envsource,
          'enhancement_source':str(args.enhancement.resolve()),'source_scene':str(src),'start_camera_source':cam_source,
          'camera_start_map':camera.tolist(),'camera_gallery_map':gallery_cam.tolist(),
          'gallery_columns':cols,'gallery_rows':nrows,'region_count':len(selected),'regions':selected,
          'seed':args.seed,'sh_degree':3,'point_sigma_px':args.point_sigma,'background':[.025,.035,.055],
          'notes':['Illustrative explosion / presentation layout, NOT physical motion or a navigation route.',
                   'Original point clouds and real Gaussian records preserved separately. No new model training.',
                   'Objects translate ONLY. No rotation or rescaling of covariances / spherical-harmonic basis.',
                   'Independent subsets can overlap; selected regions are not an exclusive complete asset inventory.',
                   'Point stage uses colored 3-D point samples rasterized as small isotropic dot sprites.',
                   'Final object stage uses original anisotropic Gaussian shapes and original SH coefficients.',
                   'Original world PLY is referenced unless --portable was requested. Sources are never rewritten.']}
    # Record ALL omissions, including unprepared or empty regions, not just showreel heuristics.
    eligible_ids={r['region_id'] for r in rows}
    source_rows=read_json(src/'scene.json')['regions']
    reviews=read_json(src/'review.json').get('regions',{}) if (src/'review.json').exists() else {}
    omitted=[]
    for item in source_rows:
        rid=item['region_id']
        if rid in eligible_ids:continue
        folder=args.enhancement.expanduser().resolve()/'regions'/rid
        if not (folder/'complete.json').exists():reason='enhancement not prepared/completed'
        elif not (folder/'gaussians_original.ply').exists():reason='no original Gaussian PLY'
        elif reviews.get(rid,{}).get('decision')=='rejected' and not getattr(args,'include_rejected',False):reason='manually rejected; explicit --include-rejected needed'
        else:reason='empty Gaussian subset or below automatic minimum (use --all-regions for nonempty small subsets)'
        omitted.append({'region_id':rid,'selected':False,'reason':reason})
    plan['selection_mode']='all_prepared_nonempty' if getattr(args,'all_regions',False) else ('explicit_ids' if args.regions else 'automatic_showreel')
    plan['include_rejected']=bool(getattr(args,'include_rejected',False))
    write_json(out/'selection_report.json',{'source_regions':len(source_rows),'candidates':len(rows),
               'selected':len(selected),'selection':omitted+log,'mode':plan['selection_mode']})
    print(f'Selection: {len(selected)} / {len(source_rows)} source regions. '
          'See selection_report.json for any omissions.',flush=True)
    write_json(out/'timeline.json',{'segments':[{'fraction':[0,.14],'action':'original room Gaussian hold'},
              {'fraction':[.14,.27],'action':'room-to-point-cloud crossfade'},
              {'fraction':[.27,.54],'action':'highlight selected regions; fade environment context'},
              {'fraction':[.31,.73],'action':'translate selected region point clouds to gallery; camera pullback'},
              {'fraction':[.73,.93],'action':'stagger point-to-original-Gaussian transition'},
              {'fraction':[.93,1],'action':'hold labelled Gaussian gallery'}]})
    with (out/'camera_path.jsonl').open('w') as f:
        for i in range(F):
            f.write(json.dumps({'frame':i,'time_s':float(t[i]),'K':K.tolist(),'camera_to_map':cams[i].tolist(),
                'camera_to_world':camera_map_to_scene(cams[i],scale,floor).tolist(),'depth_units':'not emitted by this RGB animation',
                'object_translation_map':offsets[i].tolist(),'object_translation_gaussian':(offsets[i]@ROT/scale).tolist(),
                'point_weight':cloud[i].tolist(),'splat_weight':splat[i].tolist(),
                'room_splat_weight':float(room_weight[i]),'environment_point_weight':float(env_weight[i])})+'\n')
    # Checksums cover the saved deliverables, not the mutable source directories.
    hashes={str(p.relative_to(out)):sha(p) for p in sorted(assets.rglob('*')) if p.is_file()}
    for name in ['motion.npz','camera_path.jsonl','selection_report.json','timeline.json']:hashes[name]=sha(out/name)
    plan['input_files_sha256']=hashes;plan['plan_id']=digest(plan);write_json(out/'plan.json',plan)
    contact(out,selected,rows)
    print(f'Prepared {len(selected)} regions, {F} video frames. Plan: {out}/plan.json',flush=True)
    print(f'Preview selection: {out}/selection.html\nThen run: bash semantic_animation/run.sh render --out "{out}" --preview',flush=True)


def contact(out,selected,allrows):
    cols=4;w=260;h=255;sheet=Image.new('RGB',(cols*w,math.ceil(len(selected)/cols)*h),(14,20,28));draw=ImageDraw.Draw(sheet)
    cards=[]
    for i,r in enumerate(selected):
        d=out/r['asset_dir'];e=r['evidence'];impath=d/e[0]['masked_crop'] if e and 'masked_crop' in e[0] else None
        x=i%cols*w;y=i//cols*h
        if impath and impath.exists():sheet.paste(image_fit(impath,(w-10,h-70)),(x+5,y+5))
        draw.text((x+7,y+h-60),r['display_label'][:32],font=font(14),fill=tuple(r['color']))
        draw.text((x+7,y+h-40),f'{r["gaussians"]:,} real splats | {r["source_views_count"]} views',font=font(12),fill='white')
        draw.text((x+7,y+h-23),r['region_id'][:32],font=font(11),fill=(170,185,200))
        image_rel=str(impath.relative_to(out)) if impath and impath.exists() else ''
        cards.append(f'<article><img src="{html.escape(image_rel)}"><h3>{html.escape(r["display_label"])}</h3><code>{html.escape(r["region_id"])}</code><p>{r["gaussians"]:,} original Gaussians; {r["point_count"]:,} surface points.</p><a href="{r["asset_dir"]}/asset.json">Metadata and transforms</a></article>')
    sheet.save(out/'selected_assets.jpg',quality=92)
    other=''.join(f'<tr><td>{html.escape(r["region_id"])}</td><td>{html.escape(r["label"] or "unlabelled")}</td><td>{r["gaussians"]}</td><td>{max(r["extent"]):.2f}</td></tr>' for r in allrows)
    (out/'selection.html').write_text('<!doctype html><meta charset="utf-8"><title>Animation asset selection</title><style>body{background:#111923;color:#e6edf3;font:16px sans-serif;margin:28px}main{display:flex;flex-wrap:wrap;gap:20px}article{width:260px;background:#202b38;padding:14px}img{width:240px;height:180px;object-fit:contain}a{color:#80cfff}code{font-size:11px}td{padding:7px;border-bottom:1px solid #445}h3{margin:10px 0}</style><h1>Selected region hypotheses</h1><p>Region hypotheses, not confirmed unique physical assets. --all-regions includes all prepared nonempty subsets; --regions selects explicit IDs. See selection_report.json for omissions.</p><main>'+''.join(cards)+'</main><h2>All available candidates</h2><table><tr><th>Region ID</th><th>Reviewed/suggested label</th><th>Gaussians</th><th>Max extent (m, provisional)</th></tr>'+other+'</table>')
