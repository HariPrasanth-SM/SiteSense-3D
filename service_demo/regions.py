from __future__ import annotations
import io
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read_json

class Regions:
    def __init__(self, scene, enhancement=None):
        self.root=Path(scene).resolve();self.enh=Path(enhancement).resolve() if enhancement else None
        manifest=read_json(self.root/'scene.json');self.manifest=manifest;self.items={};self.obs={};self.cache={}
        review=read_json(self.root/'review.json').get('regions',{}) if (self.root/'review.json').exists() else {}
        for item in manifest['regions']:
            rid=item['region_id'];r=read_json(self.root/'regions'/rid/'region.json');rev=review.get(rid,{})
            extra=self.enh/'regions'/rid if self.enh else None
            meta=read_json(extra/'enhancement.json') if extra and (extra/'enhancement.json').exists() else {}
            box=meta.get('clean_bbox',r['bbox']);r=dict(r, bbox=box,review=rev)
            r['label']=rev.get('semantic_label') or rid
            r['gaussian_count']=int(meta.get('selected_gaussians',meta.get('gaussian_count',0)))
            self.items[rid]=r
            for v in r['views']:self.obs.setdefault(v['observation_id'],[]).append(rid)
    def public(self):
        return [{'id':rid,'label':r['label'],'center':r['bbox']['center'],'extent':r['bbox']['extent'],
                 'rejected':r['review'].get('decision')=='rejected','kind':r['review'].get('kind','unknown'),
                 'views':len(r['views']),'point_count':r.get('point_count'),
                 'crop_url':f'/region/{rid}/0/crop','overlay_url':f'/region/{rid}/0/overlay'} for rid,r in self.items.items()]
    def cloud(self,rid):
        if rid not in self.items:raise ValueError('Unknown region')
        if rid in self.cache:return self.cache[rid]
        p=self.enh/'regions'/rid/'surface_clean.npz' if self.enh else Path('!')
        if not p.exists():p=self.root/'regions'/rid/'surface.npz'
        with np.load(p,allow_pickle=False) as d:xyz=np.asarray(d['xyz'],np.float32);rgb=d['rgb']
        if len(xyz)>7000:
            idx=np.linspace(0,len(xyz)-1,7000).astype(int);xyz=xyz[idx];rgb=rgb[idx]
        if np.issubdtype(rgb.dtype,np.floating) and rgb.max(initial=0)<=1:rgb=rgb*255
        out={'xyz':xyz.tolist(),'rgb':np.clip(rgb,0,255).astype(np.uint8).tolist(),'bbox':self.items[rid]['bbox'],'frame':'map'}
        self.cache[rid]=out;return out
    def media(self,rid,vi,kind):
        r=self.items[rid];v=r['views'][vi]
        if kind=='focus':
            from semantic_scene_plus.geometry import source_mask
            from .arrival import focused_image
            rgb=np.asarray(Image.open(Path(v['source_frame'])/'rgb.png').convert('RGB'))
            mask=source_mask(v['mask_archive'],int(v['mask_index']))
            im=Image.fromarray(focused_image(rgb,mask,tint=True));im.thumbnail((800,600))
            buf=io.BytesIO();im.save(buf,'PNG');return buf.getvalue(),'image/png'
        if kind=='original':p=Path(v['source_frame'])/'rgb.png'
        elif kind=='crop':p=self.root/'regions'/rid/v['masked_crop']
        elif kind=='context':p=self.root/'regions'/rid/v['context_crop']
        elif kind=='overlay':
            p=self.enh/'regions'/rid/'views'/f'v{vi:02d}_overlay.jpg' if self.enh else Path('!')
            if not p.exists():p=self.root/'regions'/rid/v['overview']
        elif kind=='splat':
            p=self.enh/'regions'/rid/'renders'/f'v{vi:02d}.png' if self.enh else Path('!')
            if not p.exists():raise ValueError('No pre-rendered Gaussian view exists for this object; use its cloud or source image.')
        else:raise ValueError('Unknown media kind')
        return p.read_bytes(),'image/png' if p.suffix.lower()=='.png' else 'image/jpeg'
    def detail(self,rid):
        r=self.items[rid]
        return {'id':rid,'label':r['label'],'bbox':r['bbox'],'review':r['review'],
                'views':[{'frame_id':v['frame_id'],'mask_index':v['mask_index'],'station_id':v['station_id'],
                          'crop_url':f'/region/{rid}/{i}/crop','focus_url':f'/region/{rid}/{i}/focus','overlay_url':f'/region/{rid}/{i}/overlay',
                          'original_url':f'/region/{rid}/{i}/original',
                          'splat_url':f'/region/{rid}/{i}/splat' if self.enh and (self.enh/'regions'/rid/'renders'/f'v{i:02d}.png').exists() else None} for i,v in enumerate(r['views'])]}
