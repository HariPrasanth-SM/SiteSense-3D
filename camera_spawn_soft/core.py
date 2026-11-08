"""Map-only camera station selection. No torch, ROS, rasterizer or map edits.

All distances use the existing map YAML scale. Unknown cells inside an approved
room polygon are permitted for *virtual capture only*, not navigation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, label
import yaml

VERSION = '1.0.0'


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@dataclass
class FloorMap:
    grid: np.ndarray  # row 0 is minimum local map Y
    resolution: float
    origin: np.ndarray
    identity: dict

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        cfg = yaml.safe_load(path.read_text())
        if cfg.get('mode', 'trinary') != 'trinary':
            raise ValueError('Only ROS trinary YAML maps are supported.')
        res = float(cfg['resolution'])
        org = np.asarray(cfg['origin'], dtype=float)
        if res <= 0 or not np.isfinite(res) or org.shape != (3,) or not np.isfinite(org).all():
            raise ValueError('Invalid resolution/origin in map YAML.')
        image = Path(str(cfg['image'])).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        with Image.open(image) as im:
            if im.mode in ('RGBA', 'LA', 'P'):
                raise ValueError('Use the project grayscale PGM, not a transparent/palette image.')
            a = np.asarray(im.convert('RGB'), dtype=float).mean(axis=2) / 255.
        lo, hi = float(cfg['free_thresh']), float(cfg['occupied_thresh'])
        if not 0 <= lo < hi <= 1 or int(cfg.get('negate', 0)) not in (0, 1):
            raise ValueError('Invalid trinary thresholds/negate.')
        q = a if int(cfg.get('negate', 0)) else 1 - a
        grid = np.full(q.shape, -1, dtype=np.int8)
        grid[q < lo], grid[q > hi] = 0, 100
        ident = {'yaml': str(path), 'yaml_sha256': hash_file(path),
                 'image': str(image.resolve()), 'image_sha256': hash_file(image)}
        return cls(np.flipud(grid).copy(), res, org, ident)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def rotation(self):
        c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        return np.array([[c, -s], [s, c]])

    def world_to_grid(self, xy):
        """Return (column,row), with integers on cell boundaries."""
        return (np.asarray(xy, float) - self.origin[:2]) @ self.rotation / self.resolution

    def world_to_cell(self, xy):
        c, r = np.floor(self.world_to_grid(xy)).astype(int)
        return int(r), int(c)

    def cells_to_world(self, rc):
        rc = np.asarray(rc)
        p = (rc[..., [1, 0]] + .5) * self.resolution
        return p @ self.rotation.T + self.origin[:2]

    def contains(self, rc):
        r, c = rc
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]


def check_identity(saved, fmap):
    old = saved.get('map_identity', {})
    if not old or any(old.get(k) != fmap.identity.get(k) for k in ('yaml_sha256', 'image_sha256')):
        raise ValueError('The saved region/selection belongs to a different map. Select the correct map.')


def empty_region():
    return {'polygon': [], 'corridors': [], 'pins': [], 'excluded_capture_disks': []}


def import_selection(path, fmap):
    obj = json.loads(Path(path).read_text())
    check_identity(obj, fmap)
    reg = empty_region()
    for k in ('pins', 'corridors', 'excluded_capture_disks'):
        reg[k] = obj.get(k, [])
    # Do NOT infer a room polygon from isolated point positions.
    return reg


def polygon_mask(fmap, vertices):
    """Ray casting at grid centers. Polygon is a reviewed spatial limit, not a wall."""
    vv = np.asarray(vertices, dtype=float)
    if vv.ndim != 2 or vv.shape[1] != 2 or len(vv) < 3 or not np.isfinite(vv).all():
        raise ValueError('Draw and finish a room outline with at least three vertices.')
    # Test in grid coordinates to support a rotated ROS map origin.
    vv = fmap.world_to_grid(vv)
    if abs(np.sum(vv[:, 0] * np.roll(vv[:, 1], -1) - np.roll(vv[:, 0], -1) * vv[:, 1])) < 1e-8:
        raise ValueError('Room polygon has no area.')
    rr, cc = np.indices(fmap.shape)
    x, y = cc + .5, rr + .5
    inside = np.zeros(fmap.shape, bool)
    on_edge = np.zeros(fmap.shape, bool)
    for a, b in zip(vv, np.roll(vv, -1, axis=0)):
        dx, dy = b - a
        cross = (x - a[0]) * dy - (y - a[1]) * dx
        on_edge |= ((np.abs(cross) < 1e-9) & (x >= min(a[0], b[0])-1e-9)
                    & (x <= max(a[0], b[0])+1e-9) & (y >= min(a[1], b[1])-1e-9)
                    & (y <= max(a[1], b[1])+1e-9))
        if abs(dy) > 1e-12:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0] + (y-a[1])*dx/dy)
    return inside | on_edge


def prepare_layers(fmap, region):
    inside = polygon_mask(fmap, region['polygon'])
    allowed = inside & (fmap.grid != 100)
    rr, cc = np.indices(fmap.shape)
    xy = fmap.cells_to_world(np.stack([rr, cc], axis=-1))
    for disk in region.get('excluded_capture_disks', []):
        radius = float(disk['radius'])
        centre = np.asarray([disk['x'], disk['y']], float)
        if radius <= 0 or not np.isfinite(centre).all() or not np.isfinite(radius):
            raise ValueError('Invalid exclusion disk.')
        allowed &= np.linalg.norm(xy-centre, axis=-1) > radius
    if not allowed.any():
        raise ValueError('No non-occupied cells inside the selected outline.')
    # Distance to all dark cells and the source map border. Gray is not an obstacle.
    nonblack = np.pad(fmap.grid != 100, 1, constant_values=False)
    lower = distance_transform_edt(nonblack)[1:-1, 1:-1] * fmap.resolution
    lower = np.maximum(0., lower - fmap.resolution / math.sqrt(2.))
    # Region/exclusion boundary limits relocation, but is NOT counted as a measured wall.
    return {'inside': inside, 'allowed': allowed, 'clearance_m': lower}


MOVES = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
         (-1, -1, math.sqrt(2.)), (-1, 1, math.sqrt(2.)),
         (1, -1, math.sqrt(2.)), (1, 1, math.sqrt(2.))]


def local_reachable(allowed, start, resolution, max_shift):
    """Short local search only. Never crosses dark/outside/unapproved cells or corners."""
    h, w = allowed.shape
    best = {start: 0.}
    todo = [(0., start)]
    while todo:
        d, cell = heapq.heappop(todo)
        if d > best[cell] + 1e-12:
            continue
        r, c = cell
        for dr, dc, step in MOVES:
            q = r+dr, c+dc
            if not (0 <= q[0] < h and 0 <= q[1] < w) or not allowed[q]:
                continue
            if dr and dc and not (allowed[r+dr, c] and allowed[r, c+dc]):
                continue
            nd = d + step*resolution
            if nd <= max_shift+1e-12 and nd < best.get(q, float('inf'))-1e-12:
                best[q] = nd
                heapq.heappush(todo, (nd, q))
    return best


def nudge_camera(start, allowed, clearance, resolution, preferred=.18, max_shift=.30, used=None):
    """Keep a narrow camera. Prefer a small local move that achieves the desired buffer.

    Unlike erosion, this never removes a valid point for insufficient clearance.
    Returns (cell, path-distance moved). Deduplication may return None only when
    all reachable cells are already occupied by other sample stations.
    """
    used = used if used is not None else set()
    if not allowed[start]:
        raise ValueError('Nudge start is not allowed.')
    if start not in used and clearance[start] + 1e-12 >= preferred:
        return start, 0.
    reachable = local_reachable(allowed, start, resolution, max_shift)
    candidates = [c for c in reachable if c not in used]
    if not candidates:
        return None
    # Prefer the smallest move meeting the requested clearance; no global snapping.
    good = [c for c in candidates if clearance[c]+1e-12 >= preferred]
    if good:
        cell = min(good, key=lambda c: (reachable[c], -clearance[c], c))
    else:
        # A narrow corridor remains sampled: maximize achievable local clearance.
        cell = min(candidates, key=lambda c: (-clearance[c], reachable[c], c))
    return cell, reachable[cell]


def explicit_seeds(fmap, region, spacing=.35):
    seeds = []
    for i, pin in enumerate(region.get('pins', [])):
        xy = [float(pin['x']), float(pin['y'])]
        if not np.isfinite(xy).all():
            raise ValueError('Nonfinite manual pin.')
        seeds.append((fmap.world_to_cell(xy), 'pin', f'p{i:03d}'))
    for i, line in enumerate(region.get('corridors', [])):
        vv = np.asarray(line['points'], float)
        if vv.ndim != 2 or vv.shape[1] != 2 or len(vv) < 2 or not np.isfinite(vv).all():
            raise ValueError('Each priority corridor needs two or more finite vertices.')
        seen = set()
        for a, b in zip(vv[:-1], vv[1:]):
            n = max(1, math.ceil(np.linalg.norm(b-a)/spacing))
            for t in np.linspace(0, 1, n+1):
                cell = fmap.world_to_cell(a+t*(b-a))
                if cell not in seen:
                    seen.add(cell)
                    seeds.append((cell, 'corridor', f'c{i:03d}'))
    return seeds


def generate(fmap, region, count=160, preferred=.18, max_shift=.30, corridor_spacing=.35,
             gray_fraction=.20, seed=42, scale=1.43, floor_y=1.1, camera_height=1.7):
    values = [preferred, max_shift, corridor_spacing, gray_fraction, scale, floor_y, camera_height]
    if not np.isfinite(values).all() or not 1 <= count <= 2000:
        raise ValueError('Invalid count or nonfinite settings.')
    if min(preferred, max_shift) < 0 or corridor_spacing <= 0 or not 0 <= gray_fraction <= 1:
        raise ValueError('Clearance/shift >= 0; corridor spacing > 0; gray fraction in [0,1].')
    if scale <= 0 or camera_height <= 0 or not 0 <= seed < 2**32:
        raise ValueError('Scale/height must be positive and seed must be 0..2**32-1.')
    layers = prepare_layers(fmap, region)
    allowed, clear = layers['allowed'], layers['clearance_m']
    cells = np.argwhere(allowed)
    # Random permutation is deterministic and only breaks otherwise equal sampling ties.
    rng = np.random.default_rng(seed)
    cells = cells[rng.permutation(len(cells))]
    xy = fmap.cells_to_world(cells)
    nearest = np.full(len(cells), np.inf)
    cell_index = {tuple(c): i for i, c in enumerate(cells)}
    proposed = np.zeros(len(cells), bool)
    used, records, skipped = set(), [], []
    target = min(count, len(cells))

    def add(start, source, ref=None):
        nonlocal nearest
        if not fmap.contains(start) or not allowed[start]:
            reason = ('outside_source_map' if not fmap.contains(start)
                      else 'occupied' if fmap.grid[start] == 100 else 'outside_room_or_excluded')
            skipped.append({'row': int(start[0]), 'col': int(start[1]), 'source': source, 'ref': ref, 'reason': reason})
            return False
        proposed[cell_index[start]] = True
        adjusted = nudge_camera(start, allowed, clear, fmap.resolution, preferred, max_shift, used)
        if adjusted is None:
            skipped.append({'row': int(start[0]), 'col': int(start[1]), 'source': source, 'ref': ref,
                            'reason': 'all_local_cells_already_have_stations'})
            return False
        cell, path_move = adjusted
        used.add(cell)
        before, after = fmap.cells_to_world(np.asarray([start, cell]))
        clearance = float(clear[cell])
        record = {'station_id': f's{len(records):03d}', 'x': float(after[0]), 'y': float(after[1]),
                  'row': int(cell[0]), 'col': int(cell[1]), 'selection': source,
                  'priority_ref': ref, 'map_state': int(fmap.grid[cell]),
                  'unknown_capture_only': bool(fmap.grid[cell] == -1),
                  'original_x': float(before[0]), 'original_y': float(before[1]),
                  'original_row': int(start[0]), 'original_col': int(start[1]),
                  'original_map_state': int(fmap.grid[start]),
                  'local_move_path_m': float(path_move),
                  'move_euclidean_m': float(np.linalg.norm(after-before)),
                  'clearance_before_lower_bound_m': float(clear[start]),
                  'clearance_lower_bound_m': clearance, 'preferred_clearance_m': preferred,
                  'below_preferred_clearance': clearance+1e-12 < preferred,
                  'navigation_validated': False, 'three_d_clearance_verified': False,
                  'preview_camera_height_m': camera_height,
                  'preview_gaussian_xyz': [float(after[0]/scale), float(floor_y-camera_height/scale),
                                            float(after[1]/scale)]}
        records.append(record)
        nearest = np.minimum(nearest, np.linalg.norm(xy-after, axis=1))
        nearest[cell_index[cell]] = -np.inf
        return True

    explicit = explicit_seeds(fmap, region, corridor_spacing)
    explicit = list(dict.fromkeys(explicit))
    eligible_explicit = {p[0] for p in explicit if fmap.contains(p[0]) and allowed[p[0]]}
    if len(eligible_explicit) > target:
        raise ValueError(f'{len(eligible_explicit)} explicit corridor/pin seeds exceed {target} positions. '
                         'Increase positions or corridor-spacing, or shorten the marked corridor.')
    seen = set()
    for cell, src, ref in explicit:
        if cell in seen:
            continue
        seen.add(cell)
        add(cell, src, ref)

    # Reserve some samples for meaningful GRAY patches, not isolated one-cell noise.
    gray = allowed & (fmap.grid == -1)
    components, _ = label(gray)
    sizes = np.bincount(components.ravel())
    preferred_gray = gray & (sizes[components] >= 4)
    gray_idx = preferred_gray[cells[:, 0], cells[:, 1]]
    want_gray_seeds = min(target, int(round(count*gray_fraction)))
    have_gray_seeds = sum(r['original_map_state'] == -1 for r in records)
    gray_added = 0
    while len(records) < target and have_gray_seeds+gray_added < want_gray_seeds:
        valid = gray_idx & ~proposed & (nearest != -np.inf)
        if not valid.any():
            break
        ii = np.flatnonzero(valid)
        k = int(ii[np.argmax(nearest[ii])])
        if add(tuple(cells[k]), 'gray_coverage'):
            gray_added += 1

    while len(records) < target:
        valid = ~proposed & (nearest != -np.inf)
        if not valid.any():
            # Normally unnecessary. Ensure uniqueness without falsely claiming extra capacity.
            valid = nearest != -np.inf
            if not valid.any():
                break
        ii = np.flatnonzero(valid)
        k = int(ii[np.argmax(nearest[ii])])
        add(tuple(cells[k]), 'area_coverage')

    coords = np.asarray([[r['x'], r['y']] for r in records])
    if len(coords) > 1:
        dd = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        np.fill_diagonal(dd, np.inf)
        min_spacing = float(dd.min())
    else:
        min_spacing = None
    report = {'version': VERSION, 'requested_positions': count, 'actual_positions': len(records),
              'candidate_nonblack_cells': len(cells), 'gray_camera_positions': sum(r['unknown_capture_only'] for r in records),
              'corridor_positions': sum(r['selection'] == 'corridor' for r in records),
              'nudged_positions': sum(r['local_move_path_m'] > 0 for r in records),
              'below_preferred_clearance': sum(r['below_preferred_clearance'] for r in records),
              'min_clearance_lower_bound_m': min(r['clearance_lower_bound_m'] for r in records),
              'minimum_station_spacing_m': min_spacing, 'skipped_explicit_or_duplicate_seeds': skipped,
              'policy': 'Dark blocked; green and gray inside approved room eligible. Clearance is soft. '
                        'Local shifts cannot cross blocked cells/corners. No map edits or physical-clearance claims.',
              'parameters': {'positions': count, 'preferred_clearance': preferred, 'max_shift': max_shift,
                             'corridor_spacing': corridor_spacing, 'gray_fraction': gray_fraction, 'seed': seed,
                             'scale': scale, 'floor_y': floor_y, 'camera_height': camera_height},
              'warning': 'Distances reference the already-inflated 2-D map under provisional metric scale. '
                         'A reduced-clearance camera is kept intentionally. No 3-D collision test.'}
    return records, report, layers


def preview(fmap, region, records=(), layers=None, max_size=(1080, 700), all_labels=False):
    colors = np.full((*fmap.shape, 3), [106, 113, 123], dtype=np.uint8)
    colors[fmap.grid == 100] = [34, 41, 50]
    colors[fmap.grid == 0] = [206, 229, 192]
    if layers is not None:
        colors[~layers['inside']] = (colors[~layers['inside']].astype(float)*.4).astype(np.uint8)
    h, w = fmap.shape
    factor = min(max_size[0]/w, max_size[1]/h)
    size = (max(1, int(w*factor)), max(1, int(h*factor)))
    im = Image.fromarray(np.flipud(colors)).resize(size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im)
    def px(xy):
        col, row = fmap.world_to_grid(xy)
        return float(col/w*size[0]), float((h-row)/h*size[1])
    poly = region.get('polygon', [])
    if len(poly) >= 2:
        draw.line([px(p) for p in poly] + ([px(poly[0])] if len(poly) >= 3 else []), fill='#ffffff', width=2)
    for i, line in enumerate(region.get('corridors', [])):
        pts = [px(p) for p in line['points']]
        if len(pts)>1:
            draw.line(pts, fill='#00cfc3', width=2)
    for disk in region.get('excluded_capture_disks', []):
        x, y = px([disk['x'], disk['y']])
        rad = disk['radius']/fmap.resolution * factor
        draw.ellipse((x-rad, y-rad, x+rad, y+rad), outline='#ec5b89', width=2)
    for r in records:
        x, y = px([r['x'], r['y']])
        a, b = px([r['original_x'], r['original_y']])
        if r['local_move_path_m'] > 0:
            # Arrow represents endpoint displacement, not a navigation segment.
            draw.line([(a,b),(x,y)], fill='#cba872', width=1)
        color = '#ffb24c' if r['below_preferred_clearance'] else '#1f8aff'
        draw.ellipse((x-3,y-3,x+3,y+3), fill=color, outline='white', width=1)
        if r['unknown_capture_only']:
            draw.ellipse((x-5,y-5,x+5,y+5), outline='#dc71ff', width=1)
        if all_labels:
            draw.text((x+5, y-7), r['station_id'], fill='white', stroke_width=1, stroke_fill='black')
    header = Image.new('RGB', (size[0], size[1]+74), '#ffffff')
    header.paste(im, (0,74))
    d = ImageDraw.Draw(header)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 13)
    except OSError:
        font = ImageFont.load_default()
    lines = ['CAMERA SPAWNS | blue: preferred clearance met | orange: narrow, retained',
             'Purple ring: gray capture-only | white: room outline | cyan: priority corridor',
             'No source map changes. This is not a walking route or 3-D clearance validation.']
    for i,t in enumerate(lines):
        d.text((8,5+22*i), t, font=font, fill='black')
    return header


def save_results(out, fmap, region, records, report, layers):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new empty --out directory; existing plans are not overwritten.')
    timestamp = datetime.now(timezone.utc).isoformat()
    saved_region = dict(region, map_identity=fmap.identity, created_utc=timestamp,
                        gray_inside_polygon='permitted_for_virtual_capture_only')
    write_json(out/'region.json', saved_region)
    write_json(out/'camera_positions.json', {'version': VERSION, 'map_identity': fmap.identity,
                                            'parameters': report['parameters'], 'stations': records})
    write_json(out/'spawn_report.json', report)
    keys = [k for k in records[0] if k != 'preview_gaussian_xyz'] + ['gaussian_x', 'gaussian_y', 'gaussian_z']
    with (out/'camera_positions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for r in records:
            row = {k:v for k,v in r.items() if k != 'preview_gaussian_xyz'}
            row.update(zip(('gaussian_x','gaussian_y','gaussian_z'), r['preview_gaussian_xyz']))
            writer.writerow(row)
    # Compatible input for the existing dense tool. Use --clearance 0 there,
    # otherwise that old tool reapplies the hard buffer and drops narrow cameras.
    selection = {'map_identity': fmap.identity, 'pins': [
        {'x': r['x'], 'y': r['y'], 'allow_unknown': r['unknown_capture_only'],
         'note': f"Soft spawn {r['station_id']}: capture only; clearance lower bound {r['clearance_lower_bound_m']:.3f} m; "
                 'no physical/nav validation.'} for r in records],
        'corridors': [], 'excluded_capture_disks': region.get('excluded_capture_disks', []),
        'generator': 'camera_spawn_soft/'+VERSION,
        'note': 'All final stations are pins. Downstream hard clearance must be zero to retain soft-clearance exceptions.'}
    write_json(out/'selection.json', selection)
    np.save(out/'capture_allowed_mask.npy', layers['allowed'])
    preview(fmap, region, records, layers).save(out/'camera_positions.png')
    preview(fmap, region, records, layers, (1500,1100), all_labels=True).save(out/'camera_positions_labels.png')
    return out
