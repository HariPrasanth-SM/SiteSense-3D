"""Optional gsplat rendering. No training and no package/environment changes."""
from __future__ import annotations
import base64
import io
import threading
import time
from pathlib import Path
import numpy as np
from PIL import Image
from semantic_scene.common import read_json,write_json,checked,digest
from .geometry import camera_map_to_scene


class SplatRenderer:
    """Serial CUDA calls, last selected region cached. Imported/initialized lazily."""
    def __init__(self, enhancement):
        self.out=Path(enhancement).resolve();self.cfg=read_json(self.out/'config.json')
        self.mutex=threading.RLock();self.key=None;self.params=None;self.torch=None;self.raster=None

    def _load(self,rid):
        if Path(rid).name!=rid:raise ValueError('Invalid region ID')
        folder=self.out/'regions'/rid;p=folder/'gaussians_original.ply'
        if not p.exists():raise ValueError('No selected Gaussians for this region. Inspect enhancement.json; do not confuse an empty subset with a renderer failure.')
        done=read_json(folder/'complete.json');checked(p,done['files']['gaussians_original.ply'])
        key=(rid,done['files']['gaussians_original.ply']['sha256'])
        if key==self.key:return
        if self.torch is None:
            import torch
            from gsplat import rasterization
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run nvidia-smi and your existing CUDA diagnostic. CPU cloud viewer remains available.')
            self.torch=torch;self.raster=rasterization
        from avatar_demo.render_worker import _load_scene
        self.params=None
        self.torch.cuda.empty_cache()
        self.params=_load_scene(str(p),self.torch,3)
        self.key=key

    def render(self,rid,K,camera_to_map,width,height):
        if not 32<=int(width)<=1280 or not 32<=int(height)<=960:raise ValueError('Unsupported image size')
        K=np.asarray(K,np.float32);T=np.asarray(camera_to_map,np.float32)
        if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():raise ValueError('Invalid camera matrices')
        R=T[:3,:3]
        if not np.allclose(R.T@R,np.eye(3),atol=2e-3) or np.linalg.det(R)<.99:raise ValueError('Camera rotation must be orthonormal and right handed')
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(T[3],[0,0,0,1],atol=1e-5):raise ValueError('Invalid camera intrinsics/last row')
        with self.mutex:
            started=time.perf_counter();self._load(rid);torch=self.torch;s=self.cfg['scale']
            c2g=camera_map_to_scene(T,s,self.cfg['floor_y']);w2g=np.linalg.inv(c2g).astype(np.float32)
            with torch.inference_mode():
                pixels,alpha,meta=self.raster(means=self.params[0],quats=self.params[1],scales=self.params[2],
                    opacities=self.params[3],colors=self.params[4],viewmats=torch.from_numpy(w2g[None]).to('cuda'),
                    Ks=torch.from_numpy(K[None]).to('cuda'),width=int(width),height=int(height),
                    sh_degree=3,render_mode='RGB',packed=True,near_plane=.02/s,far_plane=100./s)
                rgb=pixels[0].detach().cpu().numpy();a=alpha[0,:,:,0].detach().cpu().numpy()
                # gsplat on black background yields premultiplied RGB. PNG uses straight alpha.
                rgb=np.divide(rgb,a[:,:,None],out=np.zeros_like(rgb),where=a[:,:,None]>1e-6)
                rgba=np.concatenate([np.clip(rgb,0,1),np.clip(a[:,:,None],0,1)],axis=2)
                image=Image.fromarray(np.rint(rgba*255).astype(np.uint8),'RGBA')
                count=len(self.params[0]);del pixels,alpha,meta
            metadata={'region_id':rid,'gaussians':count,'width':int(width),'height':int(height),
                      'K':K.tolist(),'camera_to_map':T.tolist(),'camera_to_world':c2g.tolist(),
                      'render_seconds_including_load':time.perf_counter()-started,
                      'representation':'true gsplat rasterization of original selected Gaussian records; no full-scene occlusion'}
            return image,metadata

    def browser(self,rid,body):
        w=int(body.get('width',640));h=int(body.get('height',480));fov=float(body.get('fov',50))
        if not 15<=fov<=120:raise ValueError('Invalid field of view')
        T=np.asarray(body['camera_to_map'],float);f=h/(2*np.tan(np.deg2rad(fov)/2))
        K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)
        im,meta=self.render(rid,K,T,w,h);buf=io.BytesIO();im.save(buf,format='PNG')
        return {'image':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),'metadata':meta}


def render_previews(args):
    if not 1<=args.views<=32 or not 64<=args.width<=1280:raise ValueError('Invalid preview count/width')
    out=args.enhancement.expanduser().resolve();cfg=read_json(out/'config.json');src=Path(cfg['source_scene'])
    renderer=SplatRenderer(out);done=0
    for folder in sorted((out/'regions').iterdir()):
        if args.region and args.region!=folder.name:continue
        if not (folder/'complete.json').exists() or not (folder/'gaussians_original.ply').exists():continue
        if args.limit is not None and done>=args.limit:break
        region=read_json(src/'regions'/folder.name/'region.json');count=min(args.views,len(region['views']))
        for i in range(count):
            v=region['views'][i];camera=read_json(Path(v['source_frame'])/'camera.json')
            H=int(camera['height']);W=int(camera['width']);width=min(args.width,W);height=max(32,round(H*width/W))
            K=np.asarray(camera['K'],np.float32).copy();K[0]*=width/W;K[1]*=height/H
            dest=folder/'renders';dest.mkdir(exist_ok=True);p=dest/f'v{i:02d}.png';mp=dest/f'v{i:02d}.json'
            recipe={'enhancement_id':cfg['enhancement_id'],'view_index':i,'width':width,'height':height,
                    'gaussian_sha256':read_json(folder/'complete.json')['files']['gaussians_original.ply']['sha256']}
            sig=digest(recipe)
            if p.exists() and mp.exists() and read_json(mp).get('recipe_hash')==sig:continue
            image,meta=renderer.render(folder.name,K,camera['camera_to_map'],width,height)
            tmp=p.with_suffix('.tmp.png');image.save(tmp);tmp.replace(p)
            write_json(mp,{**meta,'frame_id':v['frame_id'],'mask_index':v['mask_index'],'recipe_hash':sig})
        done+=1;print(f'{folder.name}: {count} original-camera splat previews ready',flush=True)
    print(f'Processed {done} region(s). Refresh viewer to see previews.',flush=True)
