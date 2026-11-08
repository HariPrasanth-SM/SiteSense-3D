"""Reuse the finished intro; rasterize a real rotating 3D Gaussian finale."""
from __future__ import annotations
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw
from semantic_animation.plan import font
from .common import *


def quat_left(q, many, torch):
    """Hamilton product q * many, wxyz order. Left multiplication rotates world axes."""
    w=q[0];v=q[1:]
    ww=many[:,0];vv=many[:,1:]
    return torch.cat(((w*ww-(vv*v).sum(-1))[:,None],
                      w*vv+ww[:,None]*v+torch.linalg.cross(v.expand_as(vv),vv,dim=-1)),dim=-1)


class TailEngine:
    def __init__(self,base,p,selected):
        import torch
        from gsplat import rasterization
        from gsplat.cuda._wrapper import spherical_harmonics
        from semantic_animation.movie import Engine
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Check nvidia-smi and the existing working gsplat environment.')
        self.t=torch;self.raster=rasterization;self.sh=spherical_harmonics
        self.base=base;self.p=p;self.selected=set(selected);self.active=None
        self.engine=Engine(base,p);self.K=self.engine.K;self.bg=self.engine.bg
        torch.set_grad_enabled(False)

    def switch(self,global_indices):
        if self.active==global_indices:return
        self.engine.bank=None;self.engine.sections=[];self.engine.bank_kinds=frozenset()
        gc.collect();self.t.cuda.empty_cache()
        p=dict(self.p);p['regions']=[self.p['regions'][i] for i in global_indices]
        self.engine.p=p;self.engine._bank({'splats'})
        self.active=list(global_indices)

    def frame(self,motion,i):
        from scipy.spatial.transform import Rotation
        t=self.t;p=self.p
        active=[j for j,w in enumerate(motion['opacity_weight'][i]) if w>1e-6]
        self.switch(active)
        bank=self.engine.bank
        T=camera_map_to_scene(motion['camera_to_map'][i],p['scale'],p['floor_y'])
        with t.inference_mode():
            means=t.empty_like(bank[0]);quats=t.empty_like(bank[1]);scales=t.empty_like(bank[2])
            opacity=t.empty_like(bank[3]);colors=t.empty((len(means),3),device='cuda',dtype=t.float32)
            eye=T[:3,3]
            for a,b,localj,kind in self.engine.sections:
                j=active[localj];region=p['regions'][j]
                old=map_to_scene(region['center'],p['scale'],p['floor_y'])
                new=map_to_scene(motion['center_map'][i,j],p['scale'],p['floor_y'])
                angle=float(motion['yaw_rad'][i,j]);size=float(motion['uniform_scale'][i,j])
                R=scene_rotation(angle)
                # Rotation of geometry and its covariance is about the original region pivot.
                rt=t.as_tensor(R,device='cuda',dtype=t.float32)
                oldt=t.as_tensor(old,device='cuda',dtype=t.float32);newt=t.as_tensor(new,device='cuda',dtype=t.float32)
                means[a:b]=(bank[0][a:b]-oldt)@rt.T*size+newt
                qxyzw=Rotation.from_matrix(R).as_quat();qr=np.r_[qxyzw[3],qxyzw[:3]]
                qr=t.as_tensor(qr,device='cuda',dtype=t.float32)
                quats[a:b]=quat_left(qr,bank[1][a:b],t)
                scales[a:b]=bank[2][a:b]*size
                opacity[a:b]=bank[3][a:b]*float(motion['opacity_weight'][i,j])
                # Keep SH coefficients in original object axes. Inverse-transform the camera.
                # This is equivalent to rotating the SH appearance field with the object.
                local_eye=old+R.T@(eye-new)/size
                local_eye=t.as_tensor(local_eye,device='cuda',dtype=t.float32)
                for lo in range(a,b,65536):
                    hi=min(lo+65536,b)
                    directions=bank[0][lo:hi]-local_eye
                    colors[lo:hi]=t.clamp_min(self.sh(p['sh_degree'],directions,bank[4][lo:hi])+.5,0.)
            rendered,alpha,meta=self.raster(means=means,quats=quats,scales=scales,opacities=opacity,
                colors=colors,viewmats=t.as_tensor(np.linalg.inv(T)[None],device='cuda',dtype=t.float32),
                Ks=self.K,width=p['width'],height=p['height'],sh_degree=None,packed=True,render_mode='RGB',
                near_plane=.03/p['scale'],far_plane=1000000./p['scale'],backgrounds=None)
            # Retain the successful packed-mode background workaround.
            image=rendered[0]+(1-alpha[0])*self.bg.reshape(1,1,3)
            rgb=image.cpu().numpy().copy()
            del rendered,alpha,meta,image,means,quats,scales,opacity,colors
        if not np.isfinite(rgb).all():raise RuntimeError('Nonfinite rendered pixels; no completion marker written.')
        return Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))


def annotate(im,plan,p,motion,i):
    im=im.convert('RGBA');W,H=im.size;f=W/1280
    overlay=Image.new('RGBA',im.size);d=ImageDraw.Draw(overlay)
    title=str(motion['stage'][i]);sub=f'{plan["selection_count"]} hand-picked regions | Original Gaussian subsets'
    d.rectangle((0,0,W,round(94*f)),fill=(7,13,22,200))
    d.text((round(25*f),round(16*f)),title,font=font(max(16,round(25*f))),fill=(241,247,255))
    d.text((round(26*f),round(53*f)),sub,font=font(max(11,round(16*f))),fill=(174,199,219))
    d.rectangle((0,H-round(32*f),W,H),fill=(7,13,22,200))
    d.text((round(20*f),H-round(25*f)),'Presentation transforms | Partial observed regions; not complete physical assets',
           font=font(max(9,round(12*f))),fill=(183,201,215))
    if motion['time_s'][i]>plan['formation_s']*.8:
        T=motion['camera_to_map'][i];K=np.asarray(p['K']);lookup={r['region_id']:j for j,r in enumerate(p['regions'])}
        label_boxes=[]
        for k,rid in enumerate(plan['selected_ids']):
            j=lookup[rid];r=p['regions'][j];c=np.array(r['center']);R=rz(float(motion['yaw_rad'][i,j]))
            pts=(corners(r['display_bounds_min'],r['display_bounds_max'])-c)@R.T
            pts=pts*motion['uniform_scale'][i,j]+motion['center_map'][i,j]
            pc=(pts-T[:3,3])@T[:3,:3]
            if np.any(pc[:,2]<=.001):continue
            uv=pc@K.T;uv=uv[:,:2]/uv[:,2,None]
            x=(uv[:,0].min()+uv[:,0].max())/2;y=uv[:,1].max()+5*f
            if x<10 or x>W-10 or y<90*f or y>H-45*f:continue
            label=f'{k+1:02d}  '+str(r['display_label'])[:32];ft=font(max(9,round(13*f)))
            bb=d.textbbox((0,0),label,font=ft);tw=bb[2]-bb[0];th=bb[3]-bb[1]
            box=(x-tw/2-5,y-2,x+tw/2+5,y+th+7)
            if box[0]<2 or box[2]>W-2:continue
            if any(not(box[2]<b[0] or box[0]>b[2] or box[3]<b[1] or box[1]>b[3]) for b in label_boxes):continue
            label_boxes.append(box)
            d.rounded_rectangle(box,radius=4,fill=(8,16,28,215),outline=tuple(r['color'])+(230,))
            d.text((x-tw/2,y),label,font=ft,fill='white')
    return Image.alpha_composite(im,overlay).convert('RGB')


def load_showcase(out):
    out=Path(out).expanduser().resolve();p=read_json(out/'plan.json');q=dict(p);pid=q.pop('plan_id')
    if digest(q)!=pid:raise ValueError('Showcase plan edited. Use a new plan directory.')
    for name,h in p['work_files'].items():
        if sha(out/name)!=h:raise ValueError('Showcase work file changed: '+name)
    base,bp=base_plan(p['base'])
    if bp['plan_id']!=p['base_plan_id'] or sha(base/'motion.npz')!=p['base_motion_sha256']:
        raise ValueError('Base animation changed.')
    for name,h in p['gaussian_hashes'].items():
        if sha(base/name)!=h:raise ValueError('Original Gaussian asset changed: '+name)
    with np.load(out/'motion.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
    return out,p,base,bp,m


def indices_preview(p):
    n=p['intro_frames'];F=p['frame_count'];nt=p['tail_frames'];fps=p['fps']
    intro=[0,int(n*.25),int(n*.7),n-1]
    secs=[0,p['formation_s']*.5,p['formation_s'],p['formation_s']+.7,
          (nt-1)/fps*.60,(nt-1)/fps-p['pullback_s'],(nt-1)/fps-p['pullback_s']/2,(nt-1)/fps]
    return sorted(set(int(np.clip(i,0,F-1)) for i in intro+[n+round(t*fps) for t in secs]))


def completed(out,i,rid):
    mp=out/'frame_metadata'/f'{i:06d}.json'
    if not mp.exists():return False
    try:
        m=read_json(mp)
        return m.get('render_id')==rid and all((out/v/f'{i:06d}.png').exists() and
            sha(out/v/f'{i:06d}.png')==m[k] for v,k in [('frames_clean','clean_sha256'),('frames','caption_sha256')])
    except (OSError,ValueError,KeyError):return False


def render(args):
    import fcntl
    out,p,base,bp,m=load_showcase(args.out)
    baseenv=read_json(base/'render_environment.json')
    if baseenv.get('plan_id')!=bp['plan_id']:raise ValueError('Base render binding disagrees with its plan.')
    code={str(x.relative_to(Path(__file__).parent)):sha(x) for x in Path(__file__).parent.rglob('*.py')}
    deps={name:importlib.metadata.version(name) for name in ['torch','gsplat','numpy','scipy','Pillow']}
    recipe={'plan_id':p['plan_id'],'base_render_id':baseenv['render_id'],'code':code,'dependencies':deps,
            'upstream_engine_sha256':sha(Path(__file__).resolve().parent.parent/'semantic_animation/movie.py')}
    rid=digest(recipe);recipe['render_id']=rid
    with (out/'.render.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another process is writing this showcase.') from e
        envpath=out/'render_environment.json'
        if envpath.exists() and read_json(envpath)['render_id']!=rid:
            if any((out/'frame_metadata').glob('*.json')):raise ValueError('Recipe changed after rendering. Use a NEW output directory.')
        write_json(envpath,recipe)
        for name in ['frames','frames_clean','frame_metadata']:(out/name).mkdir(exist_ok=True)
        if shutil.disk_usage(out).free<512*1024**2:raise RuntimeError('Less than 512 MiB disk free.')
        indices=indices_preview(p) if args.preview else list(range(p['frame_count']))
        src_indices=np.load(out/'intro_source_indices.npy',allow_pickle=False)
        new=0;reused=0;rendered_new=0;eng=None
        try:
            for i in indices:
                if completed(out,i,rid):reused+=1;continue
                if args.limit is not None and new>=args.limit:break
                tick=time.perf_counter();destclean=out/'frames_clean'/f'{i:06d}.png';destframe=out/'frames'/f'{i:06d}.png'
                extra={}
                if i<p['intro_frames']:
                    source_i=int(src_indices[i]);sm=read_json(base/'frame_metadata'/f'{source_i:06d}.json')
                    if sm['render_id']!=baseenv['render_id']:raise ValueError('Base frame recipe mismatch.')
                    for variant,key in [('frames_clean','clean_sha256'),('frames','caption_sha256')]:
                        src=base/variant/f'{source_i:06d}.png';dest=out/variant/f'{i:06d}.png'
                        if sha(src)!=sm[key]:raise ValueError('Base frame checksum failed: '+str(src))
                        tmp=dest.with_suffix('.tmp.png');shutil.copy2(src,tmp);tmp.replace(dest)
                    action='copied original';extra={'source_frame':source_i,'source_base':str(base)}
                else:
                    j=i-p['intro_frames']
                    if eng is None:eng=TailEngine(base,bp,p['selected_ids'])
                    im=eng.frame(m,j);ann=annotate(im,p,bp,m,j)
                    for dest,image in [(destclean,im),(destframe,ann)]:
                        tmp=dest.with_suffix('.tmp.png');image.save(tmp,compress_level=3);tmp.replace(dest)
                    action='rendered showcase';rendered_new+=1;extra={'tail_frame':j,'stage':str(m['stage'][j])}
                write_json(out/'frame_metadata'/f'{i:06d}.json',{'frame':i,'time_s':i/p['fps'],'render_id':rid,
                    'clean_sha256':sha(destclean),'caption_sha256':sha(destframe),'mode':action,
                    'seconds':time.perf_counter()-tick,**extra})
                new+=1
                if action!='copied original' or new%48==0:print(f'{i+1}/{p["frame_count"]} | {action} | {time.perf_counter()-tick:.2f}s',flush=True)
        finally:
            eng=None;gc.collect()
            try:
                import torch
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except ImportError:pass
            storyboard(out,p)
        count=len(list((out/'frame_metadata').glob('*.json')))
        write_json(out/'render_summary.json',{'completed_markers':count,'expected':p['frame_count'],'new':new,
            'reused':reused,'new_gpu_frames':rendered_new,'render_id':rid})
        print(f'Complete markers: {count}/{p["frame_count"]}. Storyboard: {out}/storyboard.jpg',flush=True)


def storyboard(out,p):
    ids=[i for i in indices_preview(p) if (out/'frames'/f'{i:06d}.png').is_file()]
    if not ids:return
    w=480;h=round(w*p['height']/p['width']);sheet=Image.new('RGB',(w*2,math.ceil(len(ids)/2)*(h+25)),(16,22,31));d=ImageDraw.Draw(sheet)
    links=[]
    for k,i in enumerate(ids):
        with Image.open(out/'frames'/f'{i:06d}.png') as im:thumb=im.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        x=k%2*w;y=k//2*(h+25);sheet.paste(thumb,(x,y));d.text((x+8,y+h+4),f'{i/p["fps"]:.2f}s | frame {i}',font=font(13),fill='white')
        links.append(f'<a href="frames/{i:06d}.png"><img width="480" src="frames/{i:06d}.png"></a>')
    sheet.save(out/'storyboard.jpg',quality=92)
    (out/'preview.html').write_text('<meta charset="utf-8"><title>Showcase storyboard</title><body style="background:#111923;color:white;font:16px sans-serif"><h1>Showcase preview</h1>'+''.join(links))


def encode(args):
    out=Path(args.out).expanduser().resolve();p=read_json(out/'plan.json');env=read_json(out/'render_environment.json')
    ff=shutil.which('ffmpeg')
    if not ff:raise RuntimeError('ffmpeg missing; install only ffmpeg. Do not upgrade the CUDA environment.')
    variants=['frames','frames_clean'] if args.both else ['frames_clean' if args.clean else 'frames']
    for i in range(p['frame_count']):
        if not completed(out,i,env['render_id']):raise ValueError(f'Missing/corrupt frame {i}; finish render first.')
    for v in variants:
        dest=out/('semantic_showcase_clean.mp4' if v=='frames_clean' else 'semantic_showcase.mp4');tmp=dest.with_name(dest.stem+'.tmp.mp4')
        cmd=[ff,'-hide_banner','-loglevel','warning','-y','-framerate',str(p['fps']),'-i',str(out/v/'%06d.png'),
             '-frames:v',str(p['frame_count']),'-an','-c:v','libx264','-preset','medium','-crf',str(args.crf),
             '-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
        subprocess.run(cmd,check=True);tmp.replace(dest)
        write_json(dest.with_suffix('.json'),{'plan_id':p['plan_id'],'render_id':env['render_id'],'sha256':sha(dest),
             'duration_s':p['frame_count']/p['fps'],'frames':p['frame_count'],'audio':False,'command':cmd})
        print('Saved:',dest,flush=True)
