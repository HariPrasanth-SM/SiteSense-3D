"""Geometry in the existing map frame. No torch, ROS, model downloads or map edits."""
from __future__ import annotations
import itertools
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label as components
from scipy.spatial import cKDTree

R_GS_TO_MAP = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
         if bin(i ^ j).count('1') == 1]


def transform_record(scale, floor_y):
    T = np.eye(4)
    T[:3, :3] = scale * R_GS_TO_MAP
    T[:3, 3] = [0, 0, scale * floor_y]
    return T


def scene_to_map(xyz, scale, floor_y):
    return np.asarray(xyz) @ R_GS_TO_MAP.T * scale + [0, 0, scale * floor_y]


def map_to_scene(xyz, scale, floor_y):
    return ((np.asarray(xyz) - [0, 0, scale * floor_y]) / scale) @ R_GS_TO_MAP


def camera_map(camera):
    """Rigid camera-to-map, paired with metric-depth, not scale in rotation columns."""
    s, f = float(camera['scale']), float(camera['floor_y'])
    C = np.asarray(camera['camera_to_world'], float)
    T = np.eye(4)
    T[:3, :3] = R_GS_TO_MAP @ C[:3, :3]
    T[:3, 3] = scene_to_map(C[:3, 3], s, f)
    return T


def backproject(depth_scene, uv, camera):
    uv = np.asarray(uv, dtype=np.int32)
    K = np.asarray(camera['K'], float)
    C = np.asarray(camera['camera_to_world'], float)
    rays = np.column_stack([uv[:, 0] + .5, uv[:, 1] + .5, np.ones(len(uv))]) @ np.linalg.inv(K).T
    z = depth_scene[uv[:, 1], uv[:, 0]]
    cam = rays * z[:, None]
    scene = cam @ C[:3, :3].T + C[:3, 3]
    return scene_to_map(scene, camera['scale'], camera['floor_y'])


def project(xyz_map, camera):
    scene = map_to_scene(xyz_map, camera['scale'], camera['floor_y'])
    W = np.asarray(camera['world_to_camera'], float)
    pts = scene @ W[:3, :3].T + W[:3, 3]
    p = pts @ np.asarray(camera['K'], float).T
    valid = pts[:, 2] > 1e-5
    uv = np.full((len(pts), 2), np.nan)
    uv[valid] = p[valid, :2] / p[valid, 2, None]
    # Continuous image edges: pixel center is (column+.5,row+.5).
    return uv, valid


def box_from_points(xyz):
    lo, hi = np.min(xyz, axis=0), np.max(xyz, axis=0)
    return {'center': ((lo + hi) / 2).tolist(),
            'extent': np.maximum(hi - lo, .005).tolist(), 'yaw_deg': 0.0,
            'kind': 'axis_aligned_visible_surface_box', 'source': 'observed_surface_samples'}


def checked_box(box):
    center = np.asarray(box['center'], float)
    extent = np.asarray(box['extent'], float)
    yaw = float(box.get('yaw_deg', 0))
    if center.shape != (3,) or extent.shape != (3,) or not np.isfinite([*center, *extent, yaw]).all():
        raise ValueError('Box center/size/yaw must be finite numbers.')
    if (extent <= 0).any() or (extent > 100).any():
        raise ValueError('Box sizes must be > 0 and <= 100 provisional metres.')
    return {'center': center.tolist(), 'extent': extent.tolist(),
            'yaw_deg': (yaw + 180) % 360 - 180,
            'kind': 'upright_3d_box', 'source': 'human_adjusted'}


def box_corners(box):
    c = np.asarray(box['center'], float)
    e = np.asarray(box['extent'], float) / 2
    a = math.radians(box.get('yaw_deg', 0))
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    return (signs * e) @ R.T + c


def inside_box(points, box):
    a = math.radians(box.get('yaw_deg', 0))
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    q = (points - box['center']) @ R
    return (np.abs(q) <= np.asarray(box['extent']) / 2 + 1e-7).all(1)


def make_selection(shape, width, height):
    img = Image.new('L', (width, height), 0)
    d = ImageDraw.Draw(img)
    kind = shape.get('kind')
    if kind == 'rectangle':
        vals = np.asarray(shape.get('xyxy', []), float)
        if vals.shape != (4,) or not np.isfinite(vals).all():
            raise ValueError('Draw a valid rectangle first.')
        x1, x2 = sorted(vals[[0, 2]]); y1, y2 = sorted(vals[[1, 3]])
        x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        x2, y2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2: raise ValueError('Rectangle is too small.')
        d.rectangle((x1, y1, x2-1, y2-1), fill=255)
    elif kind == 'polygon':
        p = np.asarray(shape.get('points', []), float)
        if p.ndim != 2 or p.shape[1] != 2 or not 3 <= len(p) <= 500 or not np.isfinite(p).all():
            raise ValueError('A polygon needs at least three valid vertices.')
        p[:, 0] = np.clip(p[:, 0], 0, width - 1); p[:, 1] = np.clip(p[:, 1], 0, height - 1)
        d.polygon([tuple(v) for v in p], fill=255)
    else:
        raise ValueError('Choose rectangle or polygon selection.')
    return np.asarray(img) > 0


def support(frame, raw_mask, settings):
    dep = np.asarray(frame['depth_m'], float)
    alpha = frame['alpha']
    near, far = float(settings.get('near', .15)), float(settings.get('far', 25))
    amin = float(settings.get('alpha_min', .5))
    if not np.isfinite([near, far, amin]).all() or not 0 <= near < far <= 100 or not 0 <= amin <= 1:
        raise ValueError('Use 0 <= near < far <= 100 and alpha between 0 and 1.')
    valid = raw_mask & np.isfinite(dep) & (dep > near) & (dep < far) & np.isfinite(alpha) & (alpha >= amin)
    seed = settings.get('seed')
    seed_depth = None
    if settings.get('seed_enabled', False):
        if seed is None: raise ValueError('Click an interior point using the Depth seed tool first.')
        p = np.asarray(seed, float)
        if p.shape != (2,) or not np.isfinite(p).all(): raise ValueError('Invalid seed.')
        u, v = np.floor(p).astype(int)
        if not 0 <= u < dep.shape[1] or not 0 <= v < dep.shape[0] or not valid[v, u]:
            raise ValueError('Depth seed must be inside your outline with valid depth. Choose another pixel.')
        tol = float(settings.get('depth_tolerance', .35))
        if not np.isfinite(tol) or not .001 <= tol <= 100: raise ValueError('Depth tolerance must be positive.')
        seed_depth = float(dep[v, u])
        valid &= np.abs(dep - seed_depth) <= tol
        if settings.get('connected', True):
            ids, _ = components(valid)  # 4-connected, no diagonal bridging
            valid = ids == ids[v, u]
            valid &= ids > 0
    return valid, seed_depth


def build_draft(frame, shape, settings):
    h, w = frame['rgb'].shape[:2]
    raw = make_selection(shape, w, h)
    good, seed_depth = support(frame, raw, settings)
    vv, uu = np.nonzero(good)
    if len(uu) < 8: raise ValueError('Fewer than 8 usable pixels. Widen the outline/depth band or choose another view.')
    total = len(uu)
    # Deterministic bounded sampling for interactive CPU cleanup. All raw masks are saved.
    take = np.linspace(0, total - 1, min(total, 40000), dtype=int)
    uv_all = np.column_stack([uu[take], vv[take]]).astype(np.int32)
    xyz_all = backproject(frame['depth_scene'], uv_all, frame['camera'])
    finite = np.isfinite(xyz_all).all(1)
    uv_all, xyz_all = uv_all[finite], xyz_all[finite]
    rgb_all = frame['rgb'][uv_all[:, 1], uv_all[:, 0]]
    keep = np.ones(len(xyz_all), dtype=bool)
    if settings.get('sor', True) and len(xyz_all) >= 32:
        k = min(24, len(xyz_all) - 1)
        dist, _ = cKDTree(xyz_all).query(xyz_all, k=k+1, workers=2)
        means = dist[:, 1:].mean(1)
        std_ratio = float(settings.get('std_ratio', 2.0))
        if not .5 <= std_ratio <= 10: raise ValueError('SOR std ratio must be 0.5–10.')
        keep &= means <= means.mean() + std_ratio * means.std() + 1e-12
    xyz, uv, rgb = xyz_all[keep], uv_all[keep], rgb_all[keep]
    if len(xyz) < 8: raise ValueError('Too few points after cleanup; disable SOR or widen selection.')
    target = xyz[np.argmin(np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1))]
    aim = settings.get('aim')
    aim_source = 'observed_sample_near_surface_median'
    if aim is not None:
        p = np.asarray(aim, float)
        if p.shape != (2,) or not np.isfinite(p).all(): raise ValueError('Invalid aim point.')
        u, v = np.floor(p).astype(int)
        if not 0 <= u < w or not 0 <= v < h or not good[v, u]:
            raise ValueError('Aim point must have supported depth inside the selection.')
        target = backproject(frame['depth_scene'], [[u, v]], frame['camera'])[0]
        aim_source = 'human_clicked_depth_pixel'
    box = box_from_points(xyz)
    warnings = ['Visible surfaces only; hidden object extent is not recovered.',
                'Expected Gaussian depth and the 1.43 scene scale are provisional.']
    if raw.mean() > .5: warnings.append('Large image region: inspect for wall/floor/background leakage.')
    if len(xyz)/max(1, len(xyz_all)) < .8: warnings.append('SOR removed over 20% of sampled points; inspect thin parts.')
    if seed_depth is None: warnings.append('No depth seed: a rectangle can include background. Use a polygon or depth seed.')
    return {'shape':shape, 'settings':settings, 'raw_mask':raw, 'support_mask':good,
            'xyz':xyz.astype(np.float32), 'rgb':rgb.astype(np.uint8), 'uv':uv,
            'xyz_before_sor':xyz_all.astype(np.float32), 'rgb_before_sor':rgb_all.astype(np.uint8),
            'uv_before_sor':uv_all, 'retained_samples':np.flatnonzero(keep).astype(np.int32),
            'box':box, 'target':target.tolist(), 'aim_source':aim_source, 'box_manual':False,
            'stats':{'selection_pixels':int(raw.sum()), 'depth_supported_pixels':total,
                     'sampled_points':len(xyz_all), 'retained_points':len(xyz),
                     'sor_removed':int((~keep).sum()), 'seed_depth_m':seed_depth,
                     'provisional_camera_z_range_m':np.percentile(frame['depth_m'][good],[1,50,99]).tolist()},
            'warnings':warnings}


def draft_summary(draft, camera):
    ix = np.linspace(0, len(draft['xyz']) - 1, min(len(draft['xyz']), 5000), dtype=int)
    corners = box_corners(draft['box'])
    px, ok = project(corners, camera)
    return {'id':draft['id'], 'box':draft['box'], 'target':draft['target'],
            'box_manual':draft['box_manual'], 'stats':draft['stats'], 'warnings':draft['warnings'],
            'points':draft['xyz'][ix].tolist(), 'colors':draft['rgb'][ix].tolist(),
            'corners_map':corners.tolist(), 'corners_uv':[p.tolist() if k else None for p,k in zip(px,ok)],
            'edges':EDGES,
            'points_inside_box_fraction':float(inside_box(draft['xyz'], draft['box']).mean())}


def voxel_cloud(xyz, rgb, size=.01):
    if not len(xyz): return xyz, rgb
    key = np.floor(xyz / size).astype(np.int64)
    _, ix = np.unique(key, axis=0, return_index=True)
    ix.sort()
    return xyz[ix], rgb[ix]
