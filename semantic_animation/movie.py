"""Offline, resumable gsplat frames. No window, model training, ROS, or web service."""
from __future__ import annotations
import gc
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from PIL import Image,ImageDraw
from .common import *
from .plan import font


def phase_text(u):
    if u<.14:return '01 / The captured industrial environment','Original Gaussian-splat scene'
    if u<.27:return '02 / Reveal the geometry','Point-cloud view in the same coordinate frame'
    if u<.73:return '03 / Separate the selected regions','Observed surfaces move into an illustrative inspection layout'
    if u<.93:return '04 / Restore each region\'s appearance','Original Gaussian subsets, rendered independently'
    return 'Inspectable 3D regions','Linked to source images, masks and geometry'


def caption(image,plan,motion,i):
    im=image.copy().convert('RGBA');W,H=im.size;layer=Image.new('RGBA',im.size);d=ImageDraw.Draw(layer)
    fac=W/1280.;u=i/max(1,plan['frame_count']-1);title,subtitle=phase_text(u)
    ftitle=font(max(17,round(27*fac)));fsub=font(max(11,round(17*fac)));ftiny=font(max(9,round(13*fac)))
    d.rectangle([0,0,W,round(99*fac)],fill=(7,13,22,205))
    d.text((round(28*fac),round(18*fac)),title,font=ftitle,fill=(242,247,255,255))
    d.text((round(29*fac),round(56*fac)),subtitle,font=fsub,fill=(173,199,220,255))
    foot='Selected region hypotheses | Presentation motion, not a physical route'
    d.rectangle([0,H-round(35*fac),W,H],fill=(7,13,22,205))
    d.text((round(20*fac),H-round(26*fac)),foot,font=ftiny,fill=(185,202,214,255))
    # Labels only near the gallery state, so scene clutter is not buried in text.
    if u>.66:
        opacity=int(255*float(smooth(.66,.76,u)));T=motion['camera_to_map'][i];K=np.array(plan['K'])
        for j,r in enumerate(plan['regions']):
            pts=corners(r['display_bounds_min'],r['display_bounds_max'])+motion['translation_map'][i,j]
            pc=(pts-T[:3,3])@T[:3,:3]
            if np.any(pc[:,2]<=.01):continue
            uv=pc@K.T;uv=uv[:,:2]/uv[:,2,None]
            x=float(np.mean([uv[:,0].min(),uv[:,0].max()]));y=float(uv[:,1].max()+4*fac)
            text=r['display_label'][:35];box=d.textbbox((0,0),text,font=ftiny);tw=box[2]-box[0];th=box[3]-box[1]
            x=max(tw/2+6,min(W-tw/2-6,x));y=max(100*fac,min(H-57*fac,y))
            d.rounded_rectangle([x-tw/2-6,y-3,x+tw/2+6,y+th+7],radius=4,fill=(8,16,28,int(opacity*.86)),outline=tuple(r['color'])+(opacity,))
            d.text((x-tw/2,y-1),text,font=ftiny,fill=(240,245,250,opacity))
    return Image.alpha_composite(im,layer).convert('RGB')


class Engine:
    def __init__(self,out,plan):
        import torch
        from gsplat import rasterization
        from avatar_demo.render_worker import _load_scene
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable. Check nvidia-smi and the existing GPU diagnostic; do not reinstall the working environment blindly.')
        torch.set_grad_enabled(False);torch.set_num_threads(2)
        self.torch=torch;self.raster=rasterization;self.loader=_load_scene;self.out=out;self.p=plan
        self.room=None;self.bank=None;self.sections=[];self.bank_kinds=frozenset()
        self.K=torch.tensor([plan['K']],device='cuda',dtype=torch.float32)
        self.bg=torch.tensor([plan['background']],device='cuda',dtype=torch.float32)

    def _points(self,path,color=None):
        t=self.torch;p=self.p;xyz,rgb=read_cloud(path)
        if color is not None:rgb=np.rint(.65*np.array(color)+.35*rgb).astype(np.uint8)
        xyz=map_to_scene(xyz,p['scale'],p['floor_y']).astype(np.float32);n=len(xyz)
        sh=np.zeros((n,16,3),np.float32);sh[:,0,:]=(rgb.astype(np.float32)/255-.5)/.28209479177387814
        q=np.zeros((n,4),np.float32);q[:,0]=1
        a=(xyz,q,np.full((n,3),.001,np.float32),np.full(n,.96,np.float32),sh)
        return tuple(t.from_numpy(np.ascontiguousarray(x)).to('cuda') for x in a)

    def _bank(self, kinds):
        """Preallocate active representation layers; no all-assets torch.cat spike."""
        kinds=frozenset(kinds)
        if self.bank is not None and self.bank_kinds == kinds:return
        self.bank=None;self.sections=[];gc.collect();self.torch.cuda.empty_cache()
        t=self.torch
        sources=[]
        if 'environment' in kinds:
            path=self.out/'assets/environment_points_map.npz'
            with np.load(path,allow_pickle=False) as a:n=len(a['xyz'])
            sources.append((path,None,-1,'environment',n))
        for j,r in enumerate(self.p['regions']):
            dest=self.out/r['asset_dir']
            if 'points' in kinds:
                path=dest/'display_points_map.npz'
                with np.load(path,allow_pickle=False) as a:n=len(a['xyz'])
                sources.append((path,r['color'],j,'points',n))
            if 'splats' in kinds:
                # Counts are checked against the loader below; the plan is hash-validated.
                sources.append((dest/'gaussians_original.ply',None,j,'splats',int(r['gaussians'])))
        capacity=sum(src[-1] for src in sources)
        if capacity<=0:raise RuntimeError('No primitives in the active presentation layers.')
        # Point sprites currently carry all 16 SH bases, matching the unchanged loader.
        sh_bases=16
        bytes_per=4*(3+4+3+1+3*sh_bases)
        free,total=t.cuda.mem_get_info()
        estimate=capacity*bytes_per
        print(f'Loading active layers {sorted(kinds)}: up to {capacity:,} primitives; '
              f'base tensors ~{estimate/1024**3:.2f} GiB, free GPU {free/1024**3:.2f} GiB. '
              'Rasterizer buffers need additional memory.',flush=True)
        if estimate>free*.65:
            print('WARNING: high GPU memory pressure. No region/splat is silently discarded. '
                  'A lower image resolution reduces raster buffers, not these base tensors.',flush=True)
        bank=(t.empty((capacity,3),device='cuda',dtype=t.float32),
              t.empty((capacity,4),device='cuda',dtype=t.float32),
              t.empty((capacity,3),device='cuda',dtype=t.float32),
              t.empty((capacity,),device='cuda',dtype=t.float32),
              t.empty((capacity,sh_bases,3),device='cuda',dtype=t.float32))
        offset=0
        for path,color,j,kind,expected in sources:
            params=self.loader(str(path),t,self.p['sh_degree']) if kind=='splats' else self._points(path,color)
            n=len(params[0])
            if n>expected or offset+n>capacity:
                raise RuntimeError(f'Gaussian/point count differs from the prepared plan: {path}')
            if params[4].shape[1:]!=(sh_bases,3):
                raise RuntimeError('Expected degree-3 (16,3) appearance tensors for this animation.')
            for k in range(5):bank[k][offset:offset+n].copy_(params[k])
            self.sections.append((offset,offset+n,j,kind));offset+=n
            del params
        self.bank=tuple(a[:offset] for a in bank);self.bank_kinds=kinds
        del bank;gc.collect()
        print(f'Active bank loaded: {offset:,} primitives. Original object splats are not decimated.',flush=True)

    def _draw(self,params,c2g):
        t=self.torch;p=self.p
        with t.inference_mode():
            rendered,alpha,meta=self.raster(means=params[0],quats=params[1],scales=params[2],opacities=params[3],colors=params[4],
                viewmats=t.as_tensor(np.linalg.inv(c2g)[None],device='cuda',dtype=t.float32),Ks=self.K,
                width=p['width'],height=p['height'],sh_degree=p['sh_degree'],render_mode='RGB',packed=True,
                near_plane=.03/p['scale'],far_plane=500./p['scale'],backgrounds=None)
            # gsplat 1.5.3's packed rasterizer can infer image_dims=() from
            # [nnz,2] means and reject a documented [camera,3] background.
            # Rasterize over black, then add the missing background transmittance.
            # rendered is ALREADY alpha-weighted: do not multiply it by alpha again.
            if rendered.ndim != 4 or rendered.shape[0] != 1 or rendered.shape[-1] != 3:
                raise RuntimeError(f'Unexpected RGB shape: {tuple(rendered.shape)}')
            if tuple(alpha.shape) != tuple(rendered.shape[:-1]) + (1,):
                raise RuntimeError(f'Unexpected alpha shape: {tuple(alpha.shape)}')
            composited = rendered[0] + (1.0 - alpha[0]) * self.bg.reshape(1,1,3)
            rgb=composited.detach().cpu().numpy().copy();del rendered,alpha,meta,composited
        return rgb

    def frame(self,motion,i):
        t=self.torch;p=self.p;T=motion['camera_to_map'][i];c2g=camera_map_to_scene(T,p['scale'],p['floor_y'])
        rw=float(motion['room_splat_weight'][i]);bankrgb=None;roomrgb=None
        if rw>1e-6:
            if self.room is None:
                path=Path(p['room_ply']);path=path if path.is_absolute() else self.out/path
                print('Loading original whole-room Gaussian scene...',flush=True)
                self.room=self.loader(str(path),t,p['sh_degree'])
            roomrgb=self._draw(self.room,c2g)
        elif self.room is not None:
            self.room=None;gc.collect();t.cuda.empty_cache()
        if rw<1-1e-6:
            kinds=set()
            if float(motion['environment_point_weight'][i])>1e-6:kinds.add('environment')
            if np.any(motion['point_weight'][i]>1e-6):kinds.add('points')
            if np.any(motion['splat_weight'][i]>1e-6):kinds.add('splats')
            self._bank(kinds)
            means=self.bank[0].clone();scales=self.bank[2].clone();opacity=self.bank[3].clone()
            camera=t.as_tensor(c2g,device='cuda',dtype=t.float32)
            for a,b,j,kind in self.sections:
                if j>=0:
                    delta=motion['translation_map'][i,j]@ROT/p['scale']
                    means[a:b]+=t.as_tensor(delta,device='cuda',dtype=t.float32)
                if kind=='environment':weight=float(motion['environment_point_weight'][i])
                elif kind=='points':weight=float(motion['point_weight'][i,j])
                else:weight=float(motion['splat_weight'][i,j])
                opacity[a:b]*=weight
                if kind!='splats':
                    # Point CLOUD rasterization: isotropic, fixed-screen-size dot sprites.
                    # These synthetic display sprites are never exported as original Gaussian objects.
                    z=((means[a:b]-camera[:3,3])*camera[:3,2]).sum(1).clamp(min=.001)
                    sigma=z*p['point_sigma_px']/float(p['K'][0][0])
                    scales[a:b]=sigma[:,None].expand(-1,3)
            bankrgb=self._draw((means,self.bank[1],scales,opacity,self.bank[4]),c2g)
            del means,scales,opacity
        if roomrgb is None:rgb=bankrgb
        elif bankrgb is None:rgb=roomrgb
        else:rgb=roomrgb*rw+bankrgb*(1-rw)
        if not np.isfinite(rgb).all():raise RuntimeError(f'Nonfinite render at frame {i}; no completed frame written.')
        return Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))


def validate_plan(out):
    p=read_json(out/'plan.json');saved=p['plan_id'];base=dict(p);base.pop('plan_id')
    if digest(base)!=saved:raise ValueError('Plan was edited. Create a NEW plan instead of mixing incompatible frames.')
    for name,h in p['input_files_sha256'].items():
        if not (out/name).is_file() or sha(out/name)!=h:raise ValueError('Saved input missing/changed: '+name)
    room=Path(p['room_ply']);room=room if room.is_absolute() else out/room
    if not room.exists() or sha(room)!=p['room_sha256']:raise ValueError('Room PLY missing/changed. Use original path or create a --portable plan.')
    return p


def render(args):
    import fcntl
    import torch
    out=args.out.expanduser().resolve();p=validate_plan(out)
    code={f.name:sha(f) for f in sorted(Path(__file__).parent.glob('*.py'))}
    recipe={'plan_id':p['plan_id'],'code':code,'torch':torch.__version__,'torch_cuda':torch.version.cuda,
            'gsplat':importlib.metadata.version('gsplat'),'dtype':'float32'}
    rid=digest(recipe);recipe['render_id']=rid
    old=out/'render_environment.json'
    with (out/'.render.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError('Another process is rendering this animation.') from e
        if old.exists() and read_json(old)['render_id']!=rid:
            completed=list((out/'frame_metadata').glob('*.json')) if (out/'frame_metadata').exists() else []
            if completed:
                raise ValueError('Rendering code/runtime changed and completed frames exist. '
                                 'Create a NEW animation output; do not mix old/new frames.')
            backup=out/f'render_environment.failed_{time.time_ns()}.json'
            old.rename(backup)
            print(f'Previous attempt saved no completed frames; preserved its binding as {backup.name}.',flush=True)
        write_json(old,recipe)
        with np.load(out/'motion.npz',allow_pickle=False) as a:motion={k:a[k].copy() for k in a.files}
        indices=preview_indices(p['frame_count']) if args.preview else list(range(p['frame_count']))
        for name in ['frames','frames_clean','frame_metadata']: (out/name).mkdir(exist_ok=True)
        # Advisory only: compression sizes are unknown. Never store all video frames in RAM.
        if shutil.disk_usage(out).free<512*1024**2:raise RuntimeError('Less than 512 MiB free; make space before rendering.')
        eng=None;new=0;reused=0;started=time.perf_counter()
        try:
            for i in indices:
                name=f'{i:06d}';meta=out/'frame_metadata'/f'{name}.json'
                clean=out/'frames_clean'/f'{name}.png';frame=out/'frames'/f'{name}.png'
                if meta.exists():
                    m=read_json(meta)
                    if m.get('render_id')==rid and clean.exists() and frame.exists() and sha(clean)==m.get('clean_sha256') and sha(frame)==m.get('caption_sha256'):
                        reused+=1;continue
                if args.limit is not None and new>=args.limit:break
                if eng is None:eng=Engine(out,p)
                tick=time.perf_counter();im=eng.frame(motion,i);annotated=caption(im,p,motion,i)
                for dest,image in [(clean,im),(frame,annotated)]:
                    tmp=dest.with_suffix('.tmp.png');image.save(tmp,compress_level=3);tmp.replace(dest)
                write_json(meta,{'frame':i,'time_s':float(motion['time_s'][i]),'render_id':rid,
                    'clean_sha256':sha(clean),'caption_sha256':sha(frame),'seconds':time.perf_counter()-tick,
                    'camera_record':'camera_path.jsonl','motion_record':'motion.npz','stage':phase_text(i/max(1,p['frame_count']-1))[0]})
                new+=1;print(f'Frame {i+1}/{p["frame_count"]} | t={motion["time_s"][i]:.2f}s | {time.perf_counter()-tick:.2f}s including writes',flush=True)
        finally:
            eng=None;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        done=sum((out/'frame_metadata'/f'{i:06d}.json').exists() for i in range(p['frame_count']))
        write_json(out/'render_summary.json',{'render_id':rid,'completed_markers':done,'expected':p['frame_count'],
                   'new_this_run':new,'reused_this_run':reused,'seconds':time.perf_counter()-started,'preview':args.preview})
        storyboard(out,p)
        print(f'New={new}, reused={reused}, completed={done}/{p["frame_count"]}. Storyboard: {out}/storyboard.jpg',flush=True)


def preview_indices(F):
    return sorted(set(round(q*(F-1)) for q in [0,.10,.20,.28,.40,.55,.70,.81,.91,1]))


def storyboard(out,p):
    ids=[i for i in preview_indices(p['frame_count']) if (out/'frames'/f'{i:06d}.png').exists()]
    if not ids:return
    w=480;h=round(w*p['height']/p['width']);sheet=Image.new('RGB',(w*2,(h+25)*math.ceil(len(ids)/2)),(15,21,30));d=ImageDraw.Draw(sheet)
    for k,i in enumerate(ids):
        with Image.open(out/'frames'/f'{i:06d}.png') as im:im=im.resize((w,h),Image.Resampling.LANCZOS)
        x=k%2*w;y=k//2*(h+25);sheet.paste(im,(x,y));d.text((x+8,y+h+4),f'{i/p["fps"]:.2f} s | frame {i}',font=font(13),fill='white')
    sheet.save(out/'storyboard.jpg',quality=92)
    links=''.join(f'<a href="frames/{i:06d}.png"><img width="480" src="frames/{i:06d}.png"></a>' for i in ids)
    (out/'preview.html').write_text('<!doctype html><meta charset="utf-8"><title>Animation preview</title><body style="background:#111923;color:white;font:16px sans-serif"><h1>Animation storyboard</h1><p>Rendering uses the original room and extracted Gaussian subsets. Layout motion is illustrative.</p>'+links+'</body>')


def encode(args):
    out=args.out.expanduser().resolve();p=read_json(out/'plan.json');ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise RuntimeError('ffmpeg not found. Install just ffmpeg: sudo apt install ffmpeg. PNG frames remain available.')
    env=read_json(out/'render_environment.json');variants=['frames_clean','frames'] if args.both else ['frames_clean' if args.clean else 'frames']
    for variant in variants:
        key='clean_sha256' if variant=='frames_clean' else 'caption_sha256'
        for i in range(p['frame_count']):
            frame=out/variant/f'{i:06d}.png';mp=out/'frame_metadata'/f'{i:06d}.json'
            if not frame.exists() or not mp.exists():raise ValueError(f'Missing frame {i}; finish rendering before encoding.')
            m=read_json(mp)
            if m.get('render_id')!=env['render_id'] or sha(frame)!=m.get(key):raise ValueError(f'Frame {i} identity/checksum mismatch.')
        name='semantic_reveal_clean.mp4' if variant=='frames_clean' else 'semantic_reveal.mp4';dest=out/name
        tmp=dest.with_name(dest.stem+'.tmp.mp4')
        cmd=[ffmpeg,'-hide_banner','-loglevel','warning','-y','-framerate',str(p['fps']),'-start_number','0',
             '-i',str(out/variant/'%06d.png'),'-frames:v',str(p['frame_count']),'-an','-c:v','libx264',
             '-preset','medium','-crf',str(args.crf),'-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
        subprocess.run(cmd,check=True);tmp.replace(dest)
        write_json(dest.with_suffix('.json'),{'plan_id':p['plan_id'],'render_id':env['render_id'],'ffmpeg_command':cmd,
                   'file':name,'sha256':sha(dest),'frames':p['frame_count'],'duration_s':p['frame_count']/p['fps'],'audio':False})
        print('Saved video:',dest,flush=True)
