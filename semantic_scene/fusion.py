"""Per-hypothesis voxel grouping. No exclusive ownership across hypotheses."""
from __future__ import annotations
import numpy as np


def fuse_samples(xyz,rgb,observation_index,station_index,voxel_m=.02):
    xyz=np.asarray(xyz,np.float32);rgb=np.asarray(rgb,np.uint8)
    oi=np.asarray(observation_index,np.int32);si=np.asarray(station_index,np.int32)
    n=len(xyz)
    if xyz.shape!=(n,3) or rgb.shape!=(n,3) or oi.shape!=(n,) or si.shape!=(n,):
        raise ValueError('Sample arrays have incompatible shapes')
    if n==0 or not np.isfinite(xyz).all():raise ValueError('Empty/nonfinite samples')
    if not np.isfinite(voxel_m) or voxel_m<=0:raise ValueError('voxel_m must be positive')
    keys=np.floor(xyz.astype(float)/voxel_m).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse);m=len(count)
    # Equal observation weighting, not equal density weighting, in each voxel.
    pairs,within=np.unique(np.column_stack((inverse,oi)),axis=0,return_inverse=True)
    per_obs=np.bincount(within)
    obs_xyz=np.column_stack([np.bincount(within,weights=xyz[:,j])/per_obs for j in range(3)])
    obs_rgb=np.column_stack([np.bincount(within,weights=rgb[:,j])/per_obs for j in range(3)])
    obs_count=np.bincount(pairs[:,0],minlength=m)
    fused=np.column_stack([np.bincount(pairs[:,0],weights=obs_xyz[:,j],minlength=m)/obs_count for j in range(3)])
    colors=np.column_stack([np.bincount(pairs[:,0],weights=obs_rgb[:,j],minlength=m)/obs_count for j in range(3)])
    station_pairs=np.unique(np.column_stack((inverse,si)),axis=0)
    station_count=np.bincount(station_pairs[:,0],minlength=m)
    # Many-to-many provenance as CSR: each fused sample has a list of contributing observations.
    offsets=np.r_[0,np.cumsum(obs_count)].astype(np.int64)
    return {'xyz':fused.astype(np.float32),'rgb':np.clip(np.rint(colors),0,255).astype(np.uint8),
            'sample_count':count.astype(np.int32),'observation_count':obs_count.astype(np.int16),
            'station_count':station_count.astype(np.int16),'sample_to_voxel':inverse.astype(np.int32),
            'membership_offsets':offsets,'membership_observation_indices':pairs[:,1].astype(np.int32)}


def bounding_box(points):
    a=np.asarray(points,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)==0 or not np.isfinite(a).all():raise ValueError('Bad points for bounds')
    lo=a.min(axis=0);hi=a.max(axis=0)
    return {'min':lo.tolist(),'max':hi.tolist(),'center':((lo+hi)/2).tolist(),
            'extent':(hi-lo).tolist(),'type':'axis_aligned_bounds_of_observed_samples',
            'complete_object_boundary':False}


def select_members(record,catalog,max_views=9):
    q=int(record['query_index']);seed=catalog.rows[q]
    if record['query']['observation_id']!=seed['observation_id']:raise ValueError('Query/catalog mismatch')
    result=[q];stations={seed['station_id']}
    supported=[e for e in record['edges'] if e['status']=='supported']
    def strength(e):
        return min(e['query_to_candidate']['support_fraction'],e['candidate_to_query']['support_fraction'])*e['crop_cosine']
    for e in sorted(supported,key=lambda e:(-strength(e),e['candidate_index'])):
        i=int(e['candidate_index']);r=catalog.rows[i]
        if e['candidate_observation_id']!=r['observation_id']:raise ValueError('Edge/catalog mismatch')
        if r['station_id'] in stations:continue
        result.append(i);stations.add(r['station_id'])
        if len(result)>=max_views:break
    return result


def distinct_positions(members,catalog,min_distance=.10):
    chosen=[]
    for i in members:
        p=np.asarray(catalog.rows[i]['camera_position'],float)
        if not chosen or min(np.linalg.norm(p-q) for q in chosen)>=min_distance:chosen.append(p)
    return len(chosen)
