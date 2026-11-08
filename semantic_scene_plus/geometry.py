"""Pure CPU operations; all thresholds are in the existing provisional map metres."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from semantic_scene.fusion import bounding_box

ROT = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])

def scene_to_map(xyz, scale, floor_y):
    return np.asarray(xyz, np.float64) @ ROT.T * scale + np.array([0.,0.,scale*floor_y])

def camera_map_to_scene(T, scale, floor_y):
    T = np.asarray(T, float)
    result = np.eye(4, dtype=np.float32)
    result[:3,:3] = ROT.T @ T[:3,:3]
    result[:3,3] = ROT.T @ (T[:3,3] - [0.,0.,scale*floor_y]) / scale
    return result

def statistical_keep(xyz, neighbors=24, std_ratio=2.0):
    """Mean k-neighbor distance, excluding self; retain <= mean+ratio*std.

    Similar statistical rule to SOR; not claimed bit-identical to Open3D.
    Operates on each region independently; never picks just the largest cluster.
    """
    p = np.asarray(xyz, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError('Expected nonempty XYZ array')
    finite = np.isfinite(p).all(1)
    ids = np.flatnonzero(finite)
    keep = np.zeros(len(p),bool)
    if len(ids) < 4:
        keep[ids] = True
        return keep, {'method':'insufficient_points_finite_only','input':len(p),'kept':len(ids)}
    k = min(int(neighbors), len(ids)-1)
    # Bound the temporary distance matrix rather than querying millions x k at once.
    tree = cKDTree(p[ids]); avg = np.empty(len(ids),float)
    for begin in range(0,len(ids),100000):
        dist,_ = tree.query(p[ids[begin:begin+100000]], k=k+1, workers=2)
        avg[begin:begin+len(dist)] = dist[:,1:].mean(1)
    mean, std = float(avg.mean()), float(avg.std())
    threshold = mean + std_ratio*std
    keep[ids] = avg <= threshold + 1e-12
    return keep, {'method':'mean_kNN_distance_SOR','neighbors_excluding_self':k,
        'std_ratio':std_ratio,'mean_distance_m':mean,'std_distance_m':std,
        'threshold_m':threshold,'input':len(p),'kept':int(keep.sum()),
        'removed':int((~keep).sum()),'removed_nonfinite':int((~finite).sum())}

def write_rgb_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    xyz=np.asarray(xyz,np.float32);rgb=np.asarray(rgb,np.uint8)
    a=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for j,n in enumerate(('x','y','z')):a[n]=xyz[:,j]
    for j,n in enumerate(('red','green','blue')):a[n]=rgb[:,j]
    PlyData([PlyElement.describe(a,'vertex')],text=False,byte_order='<').write(str(path))

def source_mask(path, index):
    """Decode only the chosen mask; preserve original pixels and overlap."""
    with np.load(path,allow_pickle=False) as a:
        n,h,w=map(int,a['shape']); order=str(a['bitorder'])
        if not 0 <= index < n or order!='little':raise ValueError('Invalid mask index/bit order')
        row=a['bits'][index]
    return np.unpackbits(row,count=h*w,bitorder='little').reshape(h,w).astype(bool)

def view_vote(points_map, mask, depth_m, alpha, K, camera_to_map,
              abs_depth=.12, rel_depth=.03, mask_dilation=1, alpha_min=.5):
    from scipy.ndimage import binary_dilation
    T=np.asarray(camera_to_map,float);K=np.asarray(K,float)
    if mask.shape!=depth_m.shape or alpha.shape!=depth_m.shape:
        raise ValueError('Mask/depth/alpha dimensions differ; refusing to resize')
    pc=(np.asarray(points_map,float)-T[:3,3]) @ T[:3,:3]
    projected=pc @ K.T
    good=np.isfinite(pc).all(1)&(pc[:,2]>1e-6)
    uv=np.zeros((len(pc),2),float)
    uv[good]=projected[good,:2]/projected[good,2:3]
    # Rasterizer sample centers are (col+0.5,row+0.5): containing pixel=floor(uv).
    ix=np.floor(np.clip(uv[:,0],-1e8,1e8)).astype(np.int64)
    iy=np.floor(np.clip(uv[:,1],-1e8,1e8)).astype(np.int64)
    h,w=mask.shape;inside=good&(ix>=0)&(ix<w)&(iy>=0)&(iy<h)
    ids=np.flatnonzero(inside);hit=np.zeros(len(pc),bool)
    if mask_dilation:mask=binary_dilation(mask,iterations=mask_dilation)
    if len(ids):
        u,v=ix[ids],iy[ids];d=depth_m[v,u]
        support=np.isfinite(d)&(d>0)&np.isfinite(alpha[v,u])&(alpha[v,u]>=alpha_min)
        # Occluded/out-of-view centers are simply not supporting votes, not negative votes.
        agree=np.abs(pc[ids,2]-d)<=abs_depth+rel_depth*d
        hit[ids]=support & agree & mask[v,u]
    return hit
