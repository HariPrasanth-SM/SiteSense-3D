"""CPU-only, explicit and revisable quality heuristics. NOT calibrated confidence."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import binary_erosion, label

@dataclass(frozen=True)
class Policy:
    alpha_min: float = 0.5
    min_depth_m: float = 0.15
    max_depth_m: float = 25.0
    frame_min_usable: float = 0.15
    frame_review_usable: float = 0.70
    frame_near_reject: float = 0.90
    frame_near_review: float = 0.25
    blank_std_max: float = 1.5
    blank_lap_max: float = 0.5
    blur_lap_max: float = 12.0
    blur_gradient_max: float = 6.0
    depth_jump_abs_m: float = 0.10
    depth_jump_rel: float = 0.08
    min_mask_area: int = 32
    small_mask_area: int = 128
    min_mask_usable: float = 0.20
    good_mask_usable: float = 0.70
    max_mask_area_fraction: float = 0.85
    min_geometry_pixels: int = 32
    min_geometry_ratio: float = 0.50
    max_interior_edge_ratio: float = 0.30
    min_largest_component: float = 0.55
    min_feature_conf: float = 0.25
    interior_erosion_px: int = 1
    max_geometry_samples: int = 2048

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"Policy {key} must be finite numeric")
            if value < 0: raise ValueError(f"Policy {key} cannot be negative")
        ratios = ('alpha_min','frame_min_usable','frame_review_usable','frame_near_reject',
                  'frame_near_review','min_mask_usable','good_mask_usable','max_mask_area_fraction',
                  'min_geometry_ratio','max_interior_edge_ratio','min_largest_component','min_feature_conf')
        if any(getattr(self, k)>1 for k in ratios): raise ValueError("Policy ratios must be in [0,1]")
        if self.min_depth_m >= self.max_depth_m: raise ValueError("Bad depth interval")
        if self.frame_min_usable > self.frame_review_usable: raise ValueError("Bad frame usable interval")
        if self.frame_near_review > self.frame_near_reject: raise ValueError("Bad frame near interval")
        if self.min_mask_usable > self.good_mask_usable: raise ValueError("Bad mask usable interval")
        for k in ('min_mask_area','small_mask_area','min_geometry_pixels','interior_erosion_px','max_geometry_samples'):
            if int(getattr(self,k)) != getattr(self,k): raise ValueError(f"{k} must be an integer")
        if self.max_geometry_samples < 16: raise ValueError("Need at least 16 geometry samples")
        return self

    @classmethod
    def load(cls, path=None):
        overrides = json.loads(Path(path).read_text()) if path else {}
        return cls(**overrides).validate()


def image_features(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    return gray, lap, np.hypot(gx,gy)


def depth_support(depth, alpha, p):
    finite = np.isfinite(depth) & (depth>0) & np.isfinite(alpha) & (alpha>=p.alpha_min)
    valid = finite & (depth>=p.min_depth_m) & (depth<=p.max_depth_m)
    edges = np.zeros(depth.shape, bool)
    for axis in (0,1):
        s0=[slice(None)]*2; s1=s0.copy()
        s0[axis]=slice(None,-1); s1[axis]=slice(1,None)
        a,b = depth[tuple(s0)],depth[tuple(s1)]
        vv=valid[tuple(s0)] & valid[tuple(s1)]
        with np.errstate(invalid='ignore'):
            hit=vv & (np.abs(a-b)>np.maximum(p.depth_jump_abs_m,p.depth_jump_rel*np.minimum(a,b)))
        edges[tuple(s0)] |= hit; edges[tuple(s1)] |= hit
    return finite, valid, edges


def frame_metrics(rgb, depth, alpha, p):
    if rgb.dtype!=np.uint8 or rgb.shape!=(*depth.shape,3) or alpha.shape!=depth.shape:
        raise ValueError("RGB/depth/alpha shape or dtype mismatch; NEVER resize to hide misalignment")
    gray,lap,grad=image_features(rgb)
    finite,valid,edges=depth_support(depth,alpha,p)
    denom=max(1,int(finite.sum()))
    near=float((finite & (depth<p.min_depth_m)).sum()/denom)
    far=float((finite & (depth>p.max_depth_m)).sum()/denom)
    depth_values=depth[finite]
    m={"usable_fraction":float(valid.mean()),"covered_positive_fraction":float(finite.mean()),
       "near_fraction_of_covered":near,"far_fraction_of_covered":far,
       "alpha_mean":float(np.nanmean(np.where(np.isfinite(alpha),alpha,np.nan))) if np.isfinite(alpha).any() else None,
       "gray_std":float(gray.std()),"laplacian_variance":float(lap.var()),
       "gradient_mean":float(grad.mean()),"depth_edge_fraction":float(edges.mean()),
       "depth_percentiles_05_50_95_m":np.percentile(depth_values,[5,50,95]).tolist() if depth_values.size else None}
    reject=[]; warn=[]
    if m['usable_fraction']<p.frame_min_usable: reject.append('too_little_usable_depth')
    if near>=p.frame_near_reject: reject.append('view_mostly_inside_or_very_close_to_surface')
    if m['gray_std']<p.blank_std_max and m['laplacian_variance']<p.blank_lap_max:
        reject.append('nearly_uniform_image')
    if m['usable_fraction']<p.frame_review_usable: warn.append('partial_depth_or_render_coverage')
    if near>p.frame_near_review: warn.append('many_very_near_pixels')
    if m['laplacian_variance']<p.blur_lap_max and m['gradient_mean']<p.blur_gradient_max:
        warn.append('low_detail_or_blur_candidate')
    if far>0.25: warn.append('many_far_pixels_check_scale_or_render')
    return {**m,"status":"exclude" if reject else ("review" if warn else "keep"),
            "reasons":reject,"warnings":warn}


def validate_camera(camera, shape, frame_id=None, plan_hash=None):
    if (camera['height'],camera['width'])!=tuple(shape): raise ValueError("Camera resolution mismatch")
    if frame_id is not None and camera.get('frame_id')!=frame_id: raise ValueError('Camera frame ID mismatch')
    if plan_hash is not None and camera.get('plan_hash')!=plan_hash: raise ValueError('Camera plan hash mismatch')
    if camera.get('depth_type')!='expected_camera_z': raise ValueError('Unsupported depth convention')
    if camera.get('pixel_sampling')!={'u':'column + 0.5','v':'row + 0.5'}:
        raise ValueError('Unrecognized pixel-center convention; cannot safely back-project')
    K=np.asarray(camera['K'],float); T=np.asarray(camera['camera_to_map'],float)
    if K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError('Invalid camera matrices')
    if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-6): raise ValueError('Invalid K')
    R=T[:3,:3]
    if not np.allclose(R.T@R,np.eye(3),atol=2e-4) or not np.isclose(np.linalg.det(R),1,atol=2e-4):
        raise ValueError('camera_to_map must be rigid; do not put scale in rotation')
    if not np.allclose(T[3],[0,0,0,1]): raise ValueError('Invalid homogeneous transform')


def surface_summary(mask, depth, camera, p):
    """Partial visible-surface statistics, NOT a whole-object location/box."""
    v,u=np.nonzero(mask)
    if len(u)<p.min_geometry_pixels: return None
    ids=np.linspace(0,len(u)-1,min(len(u),p.max_geometry_samples),dtype=np.int64)
    u=u[ids]; v=v[ids]; z=depth[v,u].astype(float)
    rays=np.column_stack([u+0.5,v+0.5,np.ones(len(u))]) @ np.linalg.inv(np.asarray(camera['K'],float)).T
    rays=rays/rays[:,2:3]
    pc=rays*z[:,None]
    T=np.asarray(camera['camera_to_map'],float)
    world=pc@T[:3,:3].T+T[:3,3]
    lo,hi=np.percentile(world,[5,95],axis=0)
    centered=world-world.mean(axis=0)
    cov=centered.T@centered/max(1,len(world)-1)
    ev,vec=np.linalg.eigh(cov)
    ratio=float(max(0,ev[0])/max(1e-12,float(ev.sum())))
    return {"sample_count":len(world),"support_pixels":int(mask.sum()),
            "median_map_xyz_m":np.median(world,axis=0).tolist(),
            "p05_map_xyz_m":lo.tolist(),"p95_map_xyz_m":hi.tolist(),
            "partial_extent_p05_p95_m":(hi-lo).tolist(),
            "height_p05_p50_p95_m":np.percentile(world[:,2],[5,50,95]).tolist(),
            "planarity_ratio":ratio,"plane_normal_abs_z":float(abs(vec[2,0])),
            "not_complete_object_geometry":True}


def mask_metrics(mask, score, predicted_bbox, rgb_features, depth, finite, valid, edges, camera, p):
    gray,lap,grad=rgb_features
    area=int(mask.sum()); h,w=mask.shape
    if area==0:
        return {"area_px":0,"area_fraction":0.,"score":float(score),"bbox_xyxy":None,
                "model_bbox_xyxy":list(map(float,predicted_bbox)),"geometry":None}
    v,u=np.nonzero(mask); x0,x1=int(u.min()),int(u.max())+1; y0,y1=int(v.min()),int(v.max())+1
    roi=mask[y0:y1,x0:x1]
    comps,n=label(roi)
    largest=int(np.bincount(comps.ravel())[1:].max()) if n else 0
    interior=binary_erosion(mask,iterations=p.interior_erosion_px) if p.interior_erosion_px else mask
    interior_count=int(interior.sum())
    metric_pixels=interior if interior_count>=8 else mask
    # Thin/tiny masks keep raw masks; erosion reduces geometric eligibility, never erases archive.
    geometry_mask=interior & valid & ~edges
    nf=max(1,int((mask&finite).sum()))
    geometry=surface_summary(geometry_mask,depth,camera,p)
    data={"area_px":area,"area_fraction":area/(h*w),"score":float(score),
          "bbox_xyxy":[x0,y0,x1,y1],"model_bbox_xyxy":list(map(float,predicted_bbox)),
          "touches_image_edge":x0==0 or y0==0 or x1==w or y1==h,
          "component_count":int(n),"largest_component_fraction":largest/area,
          "usable_depth_fraction":float((mask&valid).sum()/area),
          "near_fraction_of_covered":float((mask&finite&(depth<p.min_depth_m)).sum()/nf),
          "geometry_pixels":int(geometry_mask.sum()),"geometry_fraction":float(geometry_mask.sum()/area),
          "interior_edge_fraction":float((edges&metric_pixels).sum()/max(1,int(metric_pixels.sum()))),
          "gray_std":float(gray[metric_pixels].std()),"laplacian_variance":float(lap[metric_pixels].var()),
          "gradient_mean":float(grad[metric_pixels].mean()),"geometry":geometry}
    return data


def decide_mask(m, frame, p):
    reject=[]; warn=[]; hints=[]
    if frame['status']=='exclude': reject.append('frame_excluded')
    if m['area_px']<p.min_mask_area: reject.append('too_few_mask_pixels')
    if m['area_px']:
        if m['usable_depth_fraction']<p.min_mask_usable: reject.append('insufficient_depth_support')
        if m['near_fraction_of_covered']>=p.frame_near_reject: reject.append('mask_mostly_too_near')
        if m['score']<p.min_feature_conf: warn.append('low_proposal_score')
        if m['area_px']<p.small_mask_area: warn.append('small_mask_review_do_not_assume_noise')
        if m['usable_depth_fraction']<p.good_mask_usable: warn.append('partial_depth_support')
        if m['area_fraction']>p.max_mask_area_fraction: warn.append('scene_sized_mask')
        if m['largest_component_fraction']<p.min_largest_component: warn.append('fragmented_mask')
        if m['gray_std']<p.blank_std_max or (m['laplacian_variance']<p.blur_lap_max and m['gradient_mean']<p.blur_gradient_max):
            warn.append('low_detail_or_blur_candidate')
        if m['interior_edge_fraction']>p.max_interior_edge_ratio: warn.append('many_internal_depth_jumps')
        if m.get('touches_image_edge'): hints.append('edge_truncated')
        g=m.get('geometry')
        if g and g['planarity_ratio']<0.015 and g['plane_normal_abs_z']>0.9:
            h=g['height_p05_p50_p95_m'][1]
            if abs(h)<0.20: hints.append('possible_floor_like_surface')
            elif h>2.6: hints.append('possible_high_horizontal_surface')
    status='exclude' if reject else ('review' if warn else 'keep')
    geom_ok=(status=='keep' and m.get('geometry_pixels',0)>=p.min_geometry_pixels
             and m.get('geometry_fraction',0)>=p.min_geometry_ratio
             and m.get('geometry') is not None)
    return {"status":status,"reasons":reject,"warnings":warn,"hints":hints,
            "eligible_features":status=='keep',"eligible_geometry":bool(geom_ok),
            "frame_warnings":frame.get('warnings',[]),
            "heuristic_not_ground_truth":True}
