"""Optional arrival-frame FastSAM + matching DINO descriptor and geometric gate.
No training/downloads. Never paints a target merely because an asset was selected.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


def focused_image(rgb, mask, tint=False, pad=6):
    rgb = np.asarray(rgb, np.uint8)
    mask = np.asarray(mask, bool)
    if rgb.shape[:2] != mask.shape or not mask.any():
        raise ValueError('Empty or misaligned target mask')
    y, x = np.where(mask)
    x0, x1 = max(0, int(x.min())-pad), min(rgb.shape[1], int(x.max())+1+pad)
    y0, y1 = max(0, int(y.min())-pad), min(rgb.shape[0], int(y.max())+1+pad)
    out = rgb.copy()
    out[~mask] = [238, 234, 241]
    if tint:
        out[mask] = np.rint(.60*out[mask] + .40*np.array([30,255,90])).astype(np.uint8)
        out[mask & ~binary_erosion(mask)] = [25,255,75]
    return out[y0:y1,x0:x1]


def mask_points(frame, mask, cap=1024):
    d = frame['depth_m']; a = frame['alpha']
    valid = mask & np.isfinite(d) & (d > .10) & np.isfinite(a) & (a >= .5)
    y, x = np.where(valid)
    if len(x)>cap:
        ids = np.linspace(0,len(x)-1,cap,dtype=int); x,y=x[ids],y[ids]
    if not len(x):
        return np.empty((0,3),float)
    camera=frame['camera']; K=np.asarray(camera['K'],float)
    z=d[y,x]/float(camera['scale'])
    rays=np.column_stack([x+.5,y+.5,np.ones(len(x))])@np.linalg.inv(K).T
    scene=(rays*z[:,None])@np.asarray(camera['camera_to_world'])[:3,:3].T+np.asarray(camera['camera_to_world'])[:3,3]
    s=float(camera['scale']);floor=float(camera['floor_y'])
    return np.column_stack([s*scene[:,0],s*scene[:,2],s*(floor-scene[:,1])])


class ArrivalVerifier:
    def __init__(self, config):
        self.config=config; self.model=None

    def _model(self):
        if self.model is None:
            from ultralytics import FastSAM
            path=Path(self.config.get('fastsam','FastSAM-s.pt')).expanduser()
            candidates=[path,Path('models')/path.name,Path('weights')/path.name]
            path=next((p for p in candidates if p.is_file()),None)
            if path is None:
                raise FileNotFoundError('FastSAM checkpoint not found. Pass --fastsam PATH; no automatic download is performed.')
            self.model=FastSAM(str(path.resolve()))
        return self.model

    def verify(self, frame, job, encoder):
        from feature_dataset.preprocess import mask_crop, weighted_pool
        rgb=frame['rgb']; refs=np.asarray(job['references'],np.float32)
        if refs.ndim==1:refs=refs[None]
        refs=refs[np.isfinite(refs).all(1)&(np.linalg.norm(refs,axis=1)>1e-8)]
        if not len(refs):
            return {'state':'unavailable','message':'No valid target reference descriptor; no highlight.'}
        refs/=np.maximum(np.linalg.norm(refs,axis=1,keepdims=True),1e-8)
        support=np.asarray(job.get('support_xyz',[]),float).reshape(-1,3)
        support=support[np.isfinite(support).all(1)]
        if len(support)<16:
            return {'state':'unavailable','message':'Insufficient target geometry for an arrival check; no highlight.'}
        tree=cKDTree(support)
        # Run SAM on CPU: avoid holding a third GPU model with room+DINO on an 8 GB laptop.
        # Model receives PIL RGB, not an ambiguously BGR ndarray.
        result=self._model()(Image.fromarray(rgb),device='cpu',imgsz=640,conf=.25,iou=.9,
                             max_det=100,retina_masks=True,verbose=False)[0]
        if result.masks is None:
            return {'state':'not_found','message':'No masks in the arrival view; no target highlighted.'}
        masks=result.masks.data.cpu().numpy()>0.5
        if masks.shape[1:] != rgb.shape[:2]:
            raise ValueError('FastSAM mask resolution differs from RGB; refusing to resize it blindly.')
        recipe=job['feature_run']['recipe']; candidates=[]; crops=[]; weights=[]
        tolerance=float(self.config.get('arrival_distance',.30))
        for i,mask in enumerate(masks):
            area=int(mask.sum())
            if area<32 or area>mask.size*.85:continue
            points=mask_points(frame,mask)
            if len(points)<16:continue
            dist,_=tree.query(points,k=1,workers=1)
            support_fraction=float(np.mean(dist<=tolerance))
            if support_fraction<.15:continue
            crop=mask_crop(rgb,mask,int(recipe.get('crop_size',256)),float(recipe.get('crop_padding_fraction',.08)))
            if crop is None:continue
            candidates.append({'mask_index':i,'area':area,'geometry_fraction':support_fraction,
                               'median_surface_distance_m':float(np.median(dist))})
            crops.append(crop[0]);weights.append(crop[1])
        if not candidates:
            return {'state':'not_found','message':'No visible mask has sufficient target-location support; no highlight.', 'mask_count':len(masks)}
        vectors=[]
        for begin in range(0,len(crops),4):
            patches,_=encoder.infer(np.stack(crops[begin:begin+4]))
            for j,p in enumerate(patches):
                v,ok,_,_=weighted_pool(p,weights[begin+j]);vectors.append(v[0] if bool(ok[0]) else np.zeros(p.shape[-1],np.float32))
        vectors=np.asarray(vectors); scores=np.max(vectors@refs.T,axis=1)
        for row,score in zip(candidates,scores):
            row['cosine']=float(score);row['combined_score']=float(score+.08*row['geometry_fraction'])
        candidates.sort(key=lambda a:-a['combined_score'])
        threshold=float(self.config.get('arrival_min_similarity',.72))
        eligible=[c for c in candidates if c['cosine']>=threshold]
        if not eligible:
            return {'state':'not_found','message':'No target mask passed both appearance and location gates.',
                    'threshold':threshold,'candidates':candidates}
        best=eligible[0];mask=masks[best['mask_index']]
        for other in eligible[1:]:
            if best['combined_score']-other['combined_score']>.025:break
            other_mask=masks[other['mask_index']]
            iou=float((mask&other_mask).sum()/max(1,(mask|other_mask).sum()))
            if iou<.3:
                return {'state':'ambiguous','message':'Two different masks have similar evidence; no automatic highlight.',
                        'candidates':candidates}
        overlay=rgb.copy();overlay[mask]=np.rint(.55*overlay[mask]+.45*np.array([30,255,90])).astype(np.uint8)
        overlay[mask&~binary_erosion(mask)]=[25,255,75]
        return {'state':'matched','message':'Candidate target highlighted: appearance + approximate 3D support; not verified identity.',
                'mask':mask,'overlay':overlay,'crop':focused_image(rgb,mask,True),
                'best':best,'candidates':candidates,'threshold':threshold,'geometry_tolerance_m':tolerance}
