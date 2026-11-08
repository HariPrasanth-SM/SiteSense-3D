"""Persistent spawned CUDA worker; no segmentation model is needed."""
from __future__ import annotations
import queue
import time
import traceback
import numpy as np


def render_worker(config, pose, output, events, stop):
    try:
        import torch
        from gsplat import rasterization
        from avatar_demo.render_worker import _load_scene
        from avatar_demo.core import camera_matrices
        from .geometry import camera_map, transform_record
        torch.set_num_threads(2);torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA initialization failed. Check nvidia-smi and your existing gsplat virtual environment; do not reinstall automatically.')
        events.put({'status':'loading','message':'Loading the original Gaussian scene once…'})
        scene=_load_scene(config['scene'],torch,config['sh_degree'])
        events.put({'status':'ready','message':f'Loaded {len(scene[0]):,} Gaussians.'})
        last=-1;last_at=0
        while not stop.is_set():
            with pose.get_lock(): seq,x,y,yaw,height,pitch=pose[:]
            if seq==last or time.monotonic()-last_at<1/config['render_hz']:
                stop.wait(.02);continue
            last=seq;last_at=time.monotonic()
            K,C,W=camera_matrices(x,y,yaw,height,pitch,scale=config['scale'],floor_y=config['floor_y'],
                                  width=config['width'],height_px=config['height'],fov_y_deg=config['fov'])
            with torch.inference_mode():
                p,a,meta=rasterization(means=scene[0],quats=scene[1],scales=scene[2],opacities=scene[3],colors=scene[4],
                    viewmats=torch.from_numpy(W[None]).to('cuda'),Ks=torch.from_numpy(K[None]).to('cuda'),
                    width=config['width'],height=config['height'],sh_degree=config['sh_degree'],
                    render_mode='RGB+ED',packed=True,backgrounds=None,near_plane=.03/config['scale'],far_plane=60/config['scale'])
                pixel=p[0].float().cpu().numpy();alpha=a[0,:,:,0].float().cpu().numpy().copy()
                rgb=np.rint(255*np.clip(pixel[:,:,:3],0,1)).astype(np.uint8)
                depth=pixel[:,:,3].copy().astype(np.float32)
                del p,a,meta,pixel
            cam={'K':K.tolist(),'camera_to_world':C.tolist(),'world_to_camera':W.tolist(),
                 'width':config['width'],'height':config['height'],'scale':config['scale'],'floor_y':config['floor_y'],
                 'scene_to_map_scale':config['scale'],'floor_y_scene':config['floor_y'],
                 'map_pose':{'x':x,'y':y,'yaw_rad':yaw,'camera_height_m':height,'pitch_deg':pitch},
                 'coordinate_convention':'OpenCV: +X right, +Y down, +Z forward',
                 'pixel_sampling':'column+0.5,row+0.5','depth_type':'expected_camera_z',
                 'depth_scene_units':'original_Gaussian_scene','depth_m_units':'provisional_metres',
                 'sequence':int(seq),'captured_wall_time':time.time()}
            CM=camera_map(cam);cam.update(camera_to_map=CM.tolist(),map_to_camera=np.linalg.inv(CM).tolist(),
                                         scene_to_map=transform_record(config['scale'],config['floor_y']).tolist())
            frame={'id':f'live_{int(seq):09d}','sequence':int(seq),'rgb':rgb,'depth_scene':depth,
                   'depth_m':(depth*config['scale']).astype(np.float32),'alpha':alpha,'camera':cam,
                   'render_seconds':time.monotonic()-last_at}
            try:output.put(frame,timeout=.1)
            except queue.Full:
                try:output.get_nowait()
                except queue.Empty:pass
                try:output.put(frame,timeout=.1)
                except queue.Full:pass
    except Exception as exc:
        msg=f'{type(exc).__name__}: {exc}';print('\nMANUAL LABELER RENDER ERROR\n'+traceback.format_exc(),flush=True)
        events.put({'status':'error','message':msg})
