"""Visible-surface sampling and occlusion-aware pair checks. No original splat ownership."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections import OrderedDict
from pathlib import Path
import hashlib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from .common import read_json, sha, checked_file, norm_rows


@dataclass(frozen=True)
class MatchPolicy:
    samples: int = 2048
    appearance_min: float = 0.80
    context_min: float = 0.55
    surface_tol_m: float = 0.12
    depth_abs_m: float = 0.10
    depth_rel: float = 0.03
    nn_min_fraction: float = 0.25
    projection_min_fraction: float = 0.55
    min_support_pixels: int = 16
    min_visible_fraction: float = 0.05
    max_front_conflict: float = 0.25
    mask_dilation_px: int = 1
    bounds_gate_m: float = 0.50

    def validate(self):
        for k, v in asdict(self).items():
            if not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f'Invalid policy setting {k}')
        if self.samples < 32 or self.samples > 20000 or int(self.samples) != self.samples:
            raise ValueError('samples must be an integer in [32,20000]')
        for k in ('appearance_min', 'context_min', 'nn_min_fraction', 'projection_min_fraction',
                  'min_visible_fraction', 'max_front_conflict', 'depth_rel'):
            if getattr(self, k) > 1:
                raise ValueError(f'{k} must be at most one')
        if not 1 <= self.min_support_pixels <= self.samples or int(self.min_support_pixels) != self.min_support_pixels:
            raise ValueError('Invalid minimum support count')
        if int(self.mask_dilation_px) != self.mask_dilation_px or self.mask_dilation_px > 5:
            raise ValueError('mask_dilation_px must be an integer from 0 to 5')
        if self.surface_tol_m <= 0 or self.depth_abs_m <= 0:
            raise ValueError('Geometric tolerances must be positive')
        return self


def backproject(uv, depth, camera):
    """uv = integer (column,row); sample coordinates are column+.5,row+.5."""
    uv = np.asarray(uv); z = np.asarray(depth, dtype=float)
    rays = np.column_stack([uv[:, 0]+.5, uv[:, 1]+.5, np.ones(len(uv))]) @ np.linalg.inv(np.array(camera['K'], float)).T
    rays /= rays[:, 2:3]
    pc = rays * z[:, None]
    T = np.asarray(camera['camera_to_map'], float)
    return pc @ T[:3, :3].T + T[:3, 3]


def project(points, camera):
    T = np.asarray(camera['camera_to_map'], float)
    pc = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
    p = pc @ np.asarray(camera['K'], float).T
    uv = np.full((len(pc), 2), np.nan)
    valid = np.isfinite(pc).all(axis=1) & (pc[:, 2] > 1e-6)
    uv[valid] = p[valid, :2] / p[valid, 2:3]
    return uv, pc[:, 2], valid


@dataclass
class Surface:
    index: int
    frame_id: str
    mask: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    uv: np.ndarray
    camera: dict
    crop: np.ndarray
    context: np.ndarray | None
    cls: np.ndarray | None
    support_pixels: int


class GeometryStore:
    """Small LRU caches; unpack a single mask, not every mask in every image."""
    def __init__(self, catalog, policy, max_frames=6, max_surfaces=96):
        from segment_dataset.quality import Policy
        self.catalog = catalog; self.policy = policy
        self.src = catalog.source
        self.dataset = Path(self.src['dataset']); self.seg = Path(self.src['segmentation']); self.feat = Path(self.src['features'])
        self.depth_policy = Policy.load(self.seg/'quality_policy.json')
        self.frames = OrderedDict(); self.surfaces = OrderedDict()
        self.max_frames = max_frames; self.max_surfaces = max_surfaces
        self.used_bindings = {}

    def frame(self, fid):
        if fid in self.frames:
            self.frames.move_to_end(fid); return self.frames[fid]
        from segment_dataset.storage import read_source
        from segment_dataset.quality import depth_support
        bind = self.catalog.bindings[fid]
        files = {'source_complete_sha256': self.dataset/'frames'/fid/'complete.json',
                 'mask_complete_sha256': self.seg/'frames'/fid/'complete.json',
                 'feature_complete_sha256': self.feat/'frames'/fid/'complete.json',
                 'analysis_sha256': self.seg/'analysis'/self.src['quality_policy_hash']/fid/'observations.jsonl',
                 'audit_sha256': self.seg/'audit'/f'{fid}.json'}
        for key, path in files.items():
            if sha(path) != bind[key]:
                raise ValueError(f'Source changed since catalog preparation: {path}')
        # Recheck RAW depths and alpha, not only their completion-marker filenames.
        plan_stub = {'plan_hash': self.src['capture_plan_hash'],
                     'config': {'scale': read_json(self.dataset/'plan.json')['config']['scale']}}
        rgb, depth, alpha, camera, qa, _ = read_source(self.dataset, {'frame_id': fid}, plan_stub, verify=True)
        raw = self.seg/'frames'/fid
        stamp = read_json(raw/'complete.json')
        for name in ('masks.npz', 'inference.json'):
            checked_file(raw/name, stamp['files'][name])
        with np.load(raw/'masks.npz', allow_pickle=False) as a:
            bits = a['bits'].copy(); shape = tuple(int(x) for x in a['shape'])
            if str(a['bitorder']) != 'little' or shape[1:] != depth.shape:
                raise ValueError('Mask/depth dimensions or bit order mismatch')
        ff = self.feat/'frames'/fid
        fst = read_json(ff/'complete.json'); checked_file(ff/'features.npz', fst['files']['features.npz'])
        with np.load(ff/'features.npz', allow_pickle=False) as a:
            features = {key: a[key].copy() for key in ('crop_mask_pool', 'frame_mask_pool', 'crop_cls',
                                                       'crop_valid', 'frame_valid', 'cls_valid', 'mask_indices')}
        if not np.array_equal(features['mask_indices'], np.arange(shape[0])):
            raise ValueError('Feature/mask row mismatch')
        _, valid, edges = depth_support(depth, alpha, self.depth_policy)
        item = {'rgb': rgb, 'depth': depth, 'camera': camera, 'valid': valid & ~edges,
                'bits': bits, 'shape': shape, 'features': features}
        self.frames[fid] = item; self.frames.move_to_end(fid)
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)
        self.used_bindings[fid] = bind
        return item

    def surface(self, i):
        i = int(i)
        if i in self.surfaces:
            self.surfaces.move_to_end(i); return self.surfaces[i]
        r = self.catalog.rows[i]; f = self.frame(r['frame_id']); mi = r['mask_index']; n, h, w = f['shape']
        mask = np.unpackbits(f['bits'][mi], count=h*w, bitorder='little').reshape(h, w).astype(bool)
        er = self.depth_policy.interior_erosion_px
        inner = binary_erosion(mask, iterations=er) if er else mask
        support = inner & f['valid']
        v, u = np.nonzero(support); total = len(u)
        if total < self.depth_policy.min_geometry_pixels:
            raise ValueError(f'Eligible mask no longer has enough depth support: {r["observation_id"]}')
        if total > self.policy.samples:
            # Stable independent subsampling across views; never derive a Python randomized hash.
            seed = int(hashlib.sha256(r['observation_id'].encode()).hexdigest()[:16], 16)
            keep = np.sort(np.random.default_rng(seed).choice(total, self.policy.samples, replace=False))
            v = v[keep]; u = u[keep]
        uv = np.column_stack([u, v]).astype(np.int32)
        xyz = backproject(uv, f['depth'][v, u], f['camera']).astype(np.float32)
        a = f['features']
        item = Surface(i, r['frame_id'], mask, xyz, f['rgb'][v, u].copy(), uv, f['camera'],
                       norm_rows(a['crop_mask_pool'][mi]),
                       norm_rows(a['frame_mask_pool'][mi]) if a['frame_valid'][mi] else None,
                       norm_rows(a['crop_cls'][mi]) if a['cls_valid'][mi] else None,
                       total)
        self.surfaces[i] = item; self.surfaces.move_to_end(i)
        while len(self.surfaces) > self.max_surfaces:
            self.surfaces.popitem(last=False)
        return item


def reprojection(source, target, target_frame, policy):
    """Occlusion is NOT contradiction or agreement. Only visible/depth-testable samples count."""
    uvf, z, positive = project(source.xyz, target.camera)
    h, w = target.mask.shape
    visible = positive & (uvf[:, 0] >= 0) & (uvf[:, 0] < w) & (uvf[:, 1] >= 0) & (uvf[:, 1] < h)
    idx = np.flatnonzero(visible)
    uv = np.floor(uvf[idx]).astype(np.int32)
    # Only closest source sample per target pixel; don't overcount duplicate projections.
    if len(idx):
        pix = uv[:, 1]*w+uv[:, 0]
        order = np.argsort(z[idx], kind='stable')
        _, first = np.unique(pix[order], return_index=True)
        keep = order[first]; idx = idx[keep]; uv = uv[keep]
    if len(idx):
        good = target_frame['valid'][uv[:, 1], uv[:, 0]]
        idx = idx[good]; uv = uv[good]
    observed = target_frame['depth'][uv[:, 1], uv[:, 0]] if len(idx) else np.zeros(0)
    diff = z[idx] - observed
    tol = policy.depth_abs_m + policy.depth_rel*np.abs(observed)
    match = np.abs(diff) <= tol
    occluded = diff > tol
    front_conflict = diff < -tol
    expanded = binary_dilation(target.mask, iterations=policy.mask_dilation_px) if policy.mask_dilation_px else target.mask
    inside = expanded[uv[:, 1], uv[:, 0]] if len(idx) else np.zeros(0, bool)
    hit = match & inside
    offmask = match & ~inside
    testable = match | front_conflict
    hits = int(hit.sum()); nt = int(testable.sum()); nv = len(idx)
    result = {'source_samples': len(source.xyz), 'in_frame_before_unique': int(visible.sum()),
              'usable_unique_pixels': nv, 'testable_visible_pixels': nt,
              'consistent_inside_mask': hits, 'consistent_outside_mask': int(offmask.sum()),
              'occluded_pixels': int(occluded.sum()), 'front_conflict_pixels': int(front_conflict.sum()),
              'support_fraction': hits/max(1, nt), 'testable_fraction': nt/max(1, len(source.xyz)),
              'front_conflict_fraction': int(front_conflict.sum())/max(1, nv),
              'occluded_fraction': int(occluded.sum())/max(1, nv),
              'median_abs_depth_error_m': float(np.median(np.abs(diff))) if nv else None}
    # For review overlays: 1=support,2=occluded,3=front-conflict,4=matching-depth outside mask.
    code = np.zeros(nv, np.uint8); code[hit] = 1; code[occluded] = 2; code[front_conflict] = 3; code[offmask] = 4
    debug = {'uv': uv, 'code': code, 'source_sample_index': idx.astype(np.int32)}
    return result, debug


def neighborhood_overlap(a, b, tolerance):
    ab = cKDTree(b).query(a, k=1)[0]
    ba = cKDTree(a).query(b, k=1)[0]
    return {'a_near_b_fraction': float(np.mean(ab <= tolerance)),
            'b_near_a_fraction': float(np.mean(ba <= tolerance)),
            'a_to_b_median_m': float(np.median(ab)), 'b_to_a_median_m': float(np.median(ba)),
            'tolerance_m': tolerance}


def direction_pass(r, policy):
    return (r['consistent_inside_mask'] >= policy.min_support_pixels
            and r['support_fraction'] >= policy.projection_min_fraction
            and r['testable_fraction'] >= policy.min_visible_fraction
            and r['front_conflict_fraction'] <= policy.max_front_conflict)


def decide_pair(crop, context, nn, ab, ba, policy):
    reasons = []
    appearance = crop >= policy.appearance_min and (context is None or context >= policy.context_min)
    near = min(nn['a_near_b_fraction'], nn['b_near_a_fraction']) >= policy.nn_min_fraction
    aok = direction_pass(ab, policy); bok = direction_pass(ba, policy)
    if not appearance:
        reasons.append('appearance_below_policy')
    if not near:
        reasons.append('insufficient_bidirectional_surface_overlap')
    if not aok:
        reasons.append('query_to_candidate_not_supported')
    if not bok:
        reasons.append('candidate_to_query_not_supported')
    if appearance and near and aok and bok:
        return 'supported', ['appearance_and_bidirectional_surface_and_projection_support']
    adequate_a = ab['testable_visible_pixels'] >= policy.min_support_pixels
    adequate_b = ba['testable_visible_pixels'] >= policy.min_support_pixels
    contradiction = any(r['front_conflict_fraction'] > policy.max_front_conflict
                        and r['front_conflict_pixels'] >= policy.min_support_pixels for r in (ab, ba))
    # A part can overlap a larger assembly: retain its edge without treating it as same extent.
    partial = (appearance and not contradiction and (aok or bok)
               and max(nn['a_near_b_fraction'], nn['b_near_a_fraction']) >= policy.nn_min_fraction)
    if partial:
        return 'partial_overlap', reasons + ['one_sided_or_partial_support_not_merged']
    if not adequate_a or not adequate_b:
        return 'inconclusive', reasons + ['insufficient_visible_overlap_or_occlusion']
    return 'rejected', reasons + (['visible_depth_conflict'] if contradiction else [])


def check_pair(catalog, store, q, c, appearance, selected_by):
    p = store.policy; arow = catalog.rows[q]; brow = catalog.rows[c]
    if arow['station_id'] == brow['station_id']:
        raise ValueError('This pilot requires different camera stations')
    base = {'query_index': q, 'candidate_index': c, 'query_observation_id': arow['observation_id'],
            'candidate_observation_id': brow['observation_id'], 'selected_by': selected_by,
            'crop_cosine': float(appearance), 'same_instance_verified': False,
            'semantic_label': None, 'object_id': None}
    gap = np.maximum(np.maximum(catalog.lo[q]-catalog.hi[c], catalog.lo[c]-catalog.hi[q]), 0)
    gap_norm = float(np.linalg.norm(gap)); base['partial_bounds_gap_m'] = gap_norm
    base['camera_baseline_m'] = float(np.linalg.norm(np.array(arow['camera_position'])-brow['camera_position']))
    if gap_norm > p.bounds_gate_m:
        return dict(base, status='rejected', reasons=['separated_partial_surface_bounds'],
                    geometric_gate_only=True), None
    a = store.surface(q); b = store.surface(c)
    nn = neighborhood_overlap(a.xyz, b.xyz, p.surface_tol_m)
    ab, dab = reprojection(a, b, store.frame(b.frame_id), p)
    ba, dba = reprojection(b, a, store.frame(a.frame_id), p)
    ctx = float(np.clip(a.context @ b.context, -1, 1)) if a.context is not None and b.context is not None else None
    cls = float(np.clip(a.cls @ b.cls, -1, 1)) if a.cls is not None and b.cls is not None else None
    crop = float(np.clip(a.crop @ b.crop, -1, 1))
    status, reasons = decide_pair(crop, ctx, nn, ab, ba, p)
    va = np.median(a.xyz, axis=0) - np.array(a.camera['camera_to_map'])[:3, 3]
    vb = np.median(a.xyz, axis=0) - np.array(b.camera['camera_to_map'])[:3, 3]
    denom = np.linalg.norm(va)*np.linalg.norm(vb)
    angle = float(np.degrees(np.arccos(np.clip(va@vb/max(denom, 1e-12), -1, 1))))
    return dict(base, crop_cosine=crop, context_cosine=ctx, cls_cosine=cls, surface=nn,
                query_to_candidate=ab, candidate_to_query=ba, view_angle_deg=angle,
                status=status, reasons=reasons, geometric_gate_only=False), (dab, dba)
