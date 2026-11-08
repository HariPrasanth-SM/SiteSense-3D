"""One spawned GPU process: persistent scene, serialized DINO queries and cameras."""
from __future__ import annotations
import math, queue, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from avatar_demo.core import camera_matrices

def worker(config,shared_pose,live_frames,events,jobs,stop):
    scene=None;encoder=None;verifier=None;torch=None;raster=None;last_seq=-1;last_time=0;render_error=False
    def emit(kind,**kw):events.put({'kind':kind,**kw})
    def ensure_torch():
        nonlocal torch
        if torch is None:
            import torch as t
            torch=t;torch.set_num_threads(2);torch.set_grad_enabled(False)
        return torch
    def render_pose(pose):
        nonlocal scene,raster
        t=ensure_torch()
        if not t.cuda.is_available():raise RuntimeError('CUDA unavailable. Run nvidia-smi and the existing GPU diagnostic; do not reinstall packages blindly.')
        if scene is None:
            from avatar_demo.render_worker import _load_scene
            from gsplat import rasterization
            emit('render_status',state='loading',message='Loading original Gaussian scene once…')
            scene=_load_scene(config['ply'],t,3);raster=rasterization
        x,y,yaw,height,pitch=pose
        K,c2w,w2c=camera_matrices(x,y,yaw,height,pitch,scale=config['scale'],floor_y=config['floor_y'],width=config['width'],height_px=config['height'],fov_y_deg=70)
        started=time.perf_counter()
        with t.inference_mode():
            pixels,alphas,meta=raster(means=scene[0],quats=scene[1],scales=scene[2],opacities=scene[3],colors=scene[4],
                viewmats=t.from_numpy(w2c[None]).to('cuda'),Ks=t.from_numpy(K[None]).to('cuda'),width=config['width'],height=config['height'],
                sh_degree=3,render_mode='RGB+ED',packed=True,backgrounds=None,
                near_plane=.03/config['scale'],far_plane=60/config['scale'])
            pix=pixels[0].float().cpu().numpy();alpha=alphas[0,:,:,0].float().cpu().numpy()
            rgb=np.rint(np.clip(pix[:,:,:3],0,1)*255).astype(np.uint8)
            dep=pix[:,:,3].copy().astype(np.float32)
            dep[~(np.isfinite(dep)&(dep>0)&(alpha>=.5))]=np.nan
            del pixels,alphas,meta,pix
        return {'rgb':rgb,'depth_scene':dep,'depth_m':dep*config['scale'],'alpha':alpha,
                'pose':pose,'render_seconds':time.perf_counter()-started,'time':time.time(),
                'camera':{'K':K.tolist(),'camera_to_world':c2w.tolist(),'world_to_camera':w2c.tolist(),
                          'width':config['width'],'height':config['height'],'map_pose':{'x':x,'y':y,'yaw_rad':yaw,'camera_height_m':height,'pitch_deg':pitch},
                          'depth_type':'expected_camera_z','depth_units':'depth_scene original scene units; depth_m provisional metres',
                          'pixel_sampling':'column+0.5,row+0.5','scale':config['scale'],'floor_y':config['floor_y']}}
    def ensure_encoder(run, job_id, arrival=False):
        nonlocal encoder
        from feature_dataset.model import DINOEncoder
        model=run['model']
        if encoder is None:
            emit('arrival_status' if arrival else 'query_status',id=job_id,message='Loading matching DINOv3 checkpoint…')
            device=config.get('search_device','cuda:0');precision=model.get('precision','bf16') if device!='cpu' else 'fp32'
            weights=Path(config['weights'])
            if not weights.exists():
                options=sorted(Path('models').glob('dinov3_vitb16*.pth'))
                if len(options)==1:weights=options[0]
            encoder=DINOEncoder(config['repo'],weights,model.get('backbone','dinov3_vitb16'),device,precision)
        for key in ('checkpoint_sha256','official_source_sha256','backbone','dimension'):
            if model.get(key)!=encoder.identity.get(key):
                encoder=None;raise ValueError(f'DINO query/archive identity mismatch ({key}). Use the original checkpoint and vendor code.')
        return encoder

    def encode(job):
        from feature_dataset.preprocess import mask_crop,weighted_pool
        run=job['feature_run'];recipe=run['recipe']
        if not config['no_render']:emit('render_status',state='busy',message='Camera paused briefly for DINOv3 image-query encoding.')
        enc=ensure_encoder(run,job['id'])
        with Image.open(job['file']) as im:
            im=ImageOps.exif_transpose(im).convert('RGBA');rgba=np.asarray(im)
        rgb=rgba[:,:,:3];mask=rgba[:,:,3]>127
        if not mask.any():raise ValueError('Uploaded image has an empty alpha mask.')
        result=mask_crop(rgb,mask,int(recipe.get('crop_size',256)),float(recipe.get('crop_padding_fraction',.08)))
        batch,weights,spec=result;p,c=enc.infer(batch[None]);v,ok,mass,effective=weighted_pool(p[0],weights)
        if not bool(ok[0]):raise ValueError('No valid image-query descriptor.')
        emit('query_encoded',id=job['id'],vector=v[0],preprocessing=spec,effective_patches=float(effective[0]))
        if not config['no_render']:emit('render_status',state='ready',message='Image query encoded; live camera resumes on movement.')
    emit('render_status',state='disabled' if config['no_render'] else 'starting',message='Map-only mode' if config['no_render'] else 'Camera worker starting')
    while not stop.is_set():
        job=None
        try:job=jobs.get_nowait()
        except queue.Empty:pass
        if job:
            try:
                if job['kind']=='query':encode(job)
                elif job['kind']=='target':emit('target_frame',id=job['id'],frame=render_pose(job['pose']))
                elif job['kind']=='arrival':
                    from .arrival import ArrivalVerifier
                    emit('arrival_status',id=job['id'],message='Rendering the final pose and checking target correspondence…')
                    frame=render_pose(job['pose'])
                    enc=ensure_encoder(job['feature_run'],job['id'],arrival=True)
                    if job.get('reference_image'):
                        # A user-drawn RGBA crop replaces the old synthetic region reference.
                        # Encode it only on explicit arrival verification; database features stay unchanged.
                        from feature_dataset.preprocess import mask_crop, weighted_pool
                        with Image.open(job['reference_image']) as im:rgba=np.asarray(im.convert('RGBA'))
                        refmask=rgba[:,:,3]>127
                        if not refmask.any():raise ValueError('Empty manual reference alpha mask.')
                        recipe=job['feature_run']['recipe']
                        batch,weights,_=mask_crop(rgba[:,:,:3],refmask,int(recipe.get('crop_size',256)),float(recipe.get('crop_padding_fraction',.08)))
                        tokens,_=enc.infer(batch[None]);vectors,valid,_,_=weighted_pool(tokens[0],weights)
                        if not bool(valid[0]):raise ValueError('Invalid manual reference descriptor.')
                        job=dict(job,references=vectors[0][None])
                    if verifier is None:verifier=ArrivalVerifier(config)
                    emit('render_status',state='busy',message='Arrival target verification; raw camera preserved.')
                    result=verifier.verify(frame,job,enc)
                    emit('arrival_result',id=job['id'],frame=frame,result=result)
                    emit('render_status',state='ready',message='Arrival check finished; live camera resumes on movement.')
            except Exception as exc:
                msg=f'{type(exc).__name__}: {exc}';print(msg+'\n'+traceback.format_exc(),flush=True)
                emit('job_error',id=job['id'],job_kind=job['kind'],message=msg)
            continue
        if config['no_render'] or render_error:stop.wait(.05);continue
        with shared_pose.get_lock():v=list(shared_pose[:])
        seq,x,y,yaw,height,pitch=v
        if seq==last_seq or time.monotonic()-last_time<1/max(.1,config['render_hz']):stop.wait(.015);continue
        try:
            frame=render_pose([x,y,yaw,height,pitch]);frame['sequence']=int(seq)
            # Latest frame wins; do not build a stale queue while browser is busy.
            try:live_frames.put_nowait(frame)
            except queue.Full:
                try:live_frames.get_nowait()
                except queue.Empty:pass
                try:live_frames.put_nowait(frame)
                except queue.Full:pass
            last_seq=seq;last_time=time.monotonic()
        except Exception as exc:
            render_error=True;print(traceback.format_exc(),flush=True)
            emit('render_status',state='error',message=f'{type(exc).__name__}: {exc}. Navigation remains available.')
