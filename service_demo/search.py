"""Keep-only DINO retrieval. Similarity is never presented as instance probability."""
from __future__ import annotations
import io, threading
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw
from .common import read_json

class SearchIndex:
    def __init__(self, folder, regions):
        from associate_dataset.catalog import Catalog
        self.catalog=Catalog(folder,verify=True);self.regions=regions;self.lock=threading.Lock()
        self.feature_run=read_json(Path(self.catalog.source['features'])/'run.json')
        self.dataset=Path(self.catalog.source['dataset']);self.seg=Path(self.catalog.source['segmentation'])
        # Group memberships already established upstream; never merge by cosine here.
        self.by_obs=regions.obs
    def rank(self, vector, threshold=.70, limit=200):
        q=np.asarray(vector,np.float32);q/=max(float(np.linalg.norm(q)),1e-8)
        if q.shape!=(self.catalog.vectors.shape[1],):raise ValueError('Query/archive feature dimensions differ.')
        scores=np.empty(len(self.catalog.rows),np.float32)
        for start in range(0,len(scores),8192):
            v=np.asarray(self.catalog.vectors[start:start+8192],np.float32)
            v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
            scores[start:start+len(v)]=v@q
        scores=np.clip(scores,-1,1);order=np.argsort(-scores,kind='stable')
        qualified=[int(i) for i in order if scores[i]>=threshold]
        matches=[];perframe={}
        # Do not fill every page with many overlapping masks from one frame.
        for idx in qualified:
            r=self.catalog.rows[idx];fid=r['frame_id']
            if perframe.get(fid,0)>=2:continue
            perframe[fid]=perframe.get(fid,0)+1
            regions=self.by_obs.get(r['observation_id'],[])
            rid=regions[0] if regions else None
            near=[j for j in qualified[:200] if np.linalg.norm(self.catalog.centers[j]-self.catalog.centers[idx])<=.35]
            stations=len({self.catalog.rows[j]['station_id'] for j in near})
            matches.append({'index':idx,'observation_id':r['observation_id'],'frame_id':fid,'mask_index':r['mask_index'],
                'station_id':r['station_id'],'score':float(scores[idx]),'map_xyz':r['center'],
                'bbox':{'min':r['lo'],'max':r['hi']},'region_id':rid,'possible_region_ids':regions,
                'nearby_stations':stations,'crop_url':f'/match/{idx}/crop','focus_url':f'/match/{idx}/focus','overlay_url':f'/match/{idx}/overlay',
                'note':'Appearance candidate; location is rendered-depth surface support, not a confirmed physical identity.'})
            if len(matches)>=limit:break
        return {'matches':matches,'best_score':float(scores[order[0]]) if len(order) else None,
                'threshold':threshold,'eligible_observations':len(scores),'above_threshold':len(qualified),
                'no_match':not bool(matches),'score_note':'Cosine similarity, not calibrated confidence. Several views at a similar position support consistency, not proof of identity.'}
    @lru_cache(maxsize=150)
    def media(self,index,kind):
        from semantic_scene_plus.geometry import source_mask
        r=self.catalog.rows[index]
        rgb=np.asarray(Image.open(self.dataset/'frames'/r['frame_id']/'rgb.png').convert('RGB'))
        mask=source_mask(self.seg/'frames'/r['frame_id']/'masks.npz',int(r['mask_index']))
        if mask.shape!=rgb.shape[:2]:raise ValueError('Mask/RGB mismatch')
        if kind=='focus':
            from .arrival import focused_image
            im=Image.fromarray(focused_image(rgb,mask,tint=True));im.thumbnail((800,600))
        elif kind=='overlay':
            from scipy.ndimage import binary_erosion
            out=rgb.copy();out[mask]=np.rint(.72*out[mask]+.28*np.array([18,214,169])).astype(np.uint8)
            edge=mask&~binary_erosion(mask);out[edge]=[255,210,85];im=Image.fromarray(out);im.thumbnail((640,480))
        elif kind=='crop':
            from feature_dataset.preprocess import mask_crop
            yy,xx=np.where(mask)
            if len(xx)==0:raise ValueError('Empty original mask')
            x0=max(0,int(xx.min())-6);x1=min(rgb.shape[1],int(xx.max())+7)
            y0=max(0,int(yy.min())-6);y1=min(rgb.shape[0],int(yy.max())+7)
            out=rgb.copy();out[~mask]=[124,116,104];im=Image.fromarray(out[y0:y1,x0:x1]);im.thumbnail((320,240))
        elif kind=='original':im=Image.fromarray(rgb);im.thumbnail((960,720))
        else:raise ValueError('Unknown media type')
        buf=io.BytesIO();im.save(buf,'JPEG',quality=88);return buf.getvalue()
