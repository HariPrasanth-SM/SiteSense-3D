"""Stream Stage 2/3 observations into a bounded-memory keep-only search index."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import os
import shutil
import numpy as np
from .common import read_json, lines, digest, sha, checked_file, norm_rows, write_json, module_hash


class Inputs:
    def __init__(self, dataset, segmentation, features):
        from feature_dataset.source import Source
        self.src = Source(dataset, segmentation)
        self.dataset = self.src.dataset; self.seg = self.src.seg
        self.feat = Path(features).expanduser().resolve()
        self.run = read_json(self.feat/'run.json')
        self.summary = read_json(self.feat/'summary.json')
        self.policy = self.src.policy
        if read_json(self.feat/'source.json') != self.src.binding or self.run['source'] != self.src.binding:
            raise ValueError('Stage 3 source does not match capture/segmentation paths and IDs.')
        if self.summary['run_id'] != self.run['run_id']:
            raise ValueError('Feature run/summary mismatch')
        if self.summary['quality_policy_hash'] != self.policy:
            raise ValueError('Stage 3 selection is stale. Finish Stage 2 reselect and Stage 3 reindex first.')
        c = self.summary['counts']
        if c.get('extracted_frames') != c.get('segmented_frames') or c.get('remaining_frames', 0):
            raise ValueError('Complete Stage 3 extraction before preparing the full association catalog.')
        self.binding = {'schema': 'valmet.stage4.source.v1', 'dataset': str(self.dataset),
                        'segmentation': str(self.seg), 'features': str(self.feat),
                        'capture_plan_hash': self.src.plan['plan_hash'],
                        'segmentation_run_id': self.src.seg_run['run_id'],
                        'feature_run_id': self.run['run_id'], 'quality_policy_hash': self.policy,
                        'plan_file_sha256': sha(self.dataset/'plan.json'),
                        'feature_run_sha256': sha(self.feat/'run.json'),
                        'feature_summary_sha256': sha(self.feat/'summary.json')}


def prepare(inputs, out):
    out = Path(out)
    config = {'schema': 'valmet.stage4.catalog.v1', 'source': inputs.binding,
              'descriptor': 'crop_mask_pool', 'selection': 'frame keep AND mask keep AND eligible_geometry AND valid crop',
              'code_sha256': module_hash()}
    config['catalog_id'] = digest(config)
    if (out/'source.json').exists() and read_json(out/'source.json') != inputs.binding:
        raise ValueError('Different input/policy at output. Use a new Stage 4 output directory.')
    if (out/'catalog_complete.json').exists():
        done = read_json(out/'catalog_complete.json')
        if done['config'] != config:
            raise ValueError('Catalog recipe changed; use a new output.')
        for name, entry in done['files'].items():
            checked_file(out/name, entry)
        print(f"Catalog already complete: {done['count']:,} strict geometry observations.", flush=True)
        return done
    if shutil.disk_usage(out).free < 2 * 1024**3:
        raise RuntimeError('Need at least 2 GiB free for this catalog and initial reports.')
    write_json(out/'source.json', inputs.binding)
    n_expected = int(inputs.summary['counts'].get('geometry_candidates', 0))
    if n_expected < 1:
        raise ValueError('No strict geometry candidates. Do not silently enable review masks.')
    dim = int(inputs.run['model']['dimension'])
    # Only the crop descriptor is copied. Dense image features and the model are NOT loaded.
    vec_tmp = out/'crop_vectors.working.npy'
    vec = np.lib.format.open_memmap(vec_tmp, mode='w+', dtype=np.float16, shape=(n_expected, dim))
    rows_tmp = out/'catalog.working.jsonl'
    index = 0; frame_count = 0; bindings = {}; counts = Counter()
    try:
        with rows_tmp.open('w') as fo:
            for j, view in enumerate(inputs.src.frames, 1):
                fid = view['frame_id']
                rec = inputs.src.inspect(view, verify=True)
                if not rec['segmented'] or rec['frame_status'] != 'keep':
                    counts['skipped_frame_not_keep_or_no_masks'] += 1
                    continue
                folder = inputs.feat/'frames'/fid
                stamp = read_json(folder/'complete.json')
                if stamp['run_id'] != inputs.run['run_id'] or stamp['input_hash'] != rec['input_hash']:
                    raise ValueError(f'Feature input binding mismatch: {fid}')
                for name in ('features.npz', 'metadata.json'):
                    checked_file(folder/name, stamp['files'][name])
                # Depth is rechecked when actually used for an edge. Camera/RGB/mask checked above.
                capture_stamp = read_json(inputs.dataset/'frames'/fid/'complete.json')
                cam = read_json(inputs.dataset/'frames'/fid/'camera.json')
                rows = inputs.src.read_rows(fid)
                with np.load(folder/'features.npz', allow_pickle=False) as a:
                    ids = a['mask_indices']; crop = a['crop_mask_pool']; ok = a['crop_valid']
                    if not np.array_equal(ids, np.arange(len(rows))) or crop.shape != (len(rows), dim):
                        raise ValueError(f'Invalid feature row shape/order: {fid}')
                    for i, row in enumerate(rows):
                        d = row['decision']; g = row['metrics'].get('geometry')
                        eligible = (d['status'] == 'keep' and bool(d['eligible_features'])
                                    and bool(d['eligible_geometry']) and bool(ok[i]) and g is not None)
                        if not eligible:
                            continue
                        if index >= n_expected:
                            raise ValueError('Geometry count exceeds Stage 3 summary. Reindex upstream.')
                        xyz = np.asarray(g['median_map_xyz_m'], float)
                        lo = np.asarray(g['p05_map_xyz_m'], float); hi = np.asarray(g['p95_map_xyz_m'], float)
                        if xyz.shape != (3,) or not np.isfinite([xyz, lo, hi]).all() or np.any(hi < lo):
                            raise ValueError(f'Invalid Stage 2 surface summary: {fid}, {i}')
                        vec[index] = norm_rows(crop[i])
                        m = row['metrics']
                        r = {'index': index, 'observation_id': row['observation_id'], 'frame_id': fid,
                             'station_id': view['station_id'], 'mask_index': i,
                             'center': xyz.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(),
                             'area_px': int(m['area_px']), 'area_fraction': float(m['area_fraction']),
                             'bbox_xyxy': m['bbox_xyxy'], 'geometry_pixels': int(m['geometry_pixels']),
                             'geometry_fraction': float(m['geometry_fraction']),
                             'laplacian_variance': float(m['laplacian_variance']),
                             'proposal_score': float(m['score']), 'hints': d.get('hints', []),
                             'crop_effective_patches': float(a['crop_effective_patches'][i]),
                             'frame_status': 'keep', 'mask_status': 'keep', 'eligible_geometry': True,
                             'camera_position': np.asarray(cam['camera_to_map'])[:3, 3].tolist(),
                             'semantic_label': None, 'object_id': None}
                        fo.write(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n')
                        index += 1
                bindings[fid] = {'source_complete_sha256': sha(inputs.dataset/'frames'/fid/'complete.json'),
                                 'mask_complete_sha256': sha(inputs.seg/'frames'/fid/'complete.json'),
                                 'feature_complete_sha256': sha(folder/'complete.json'),
                                 'analysis_sha256': sha(inputs.seg/'analysis'/inputs.policy/fid/'observations.jsonl'),
                                 'audit_sha256': sha(inputs.seg/'audit'/f'{fid}.json')}
                frame_count += 1
                if j % 100 == 0 or j == len(inputs.src.frames):
                    print(f'Catalog {j}/{len(inputs.src.frames)} frames | {index:,} kept geometry observations', flush=True)
        if index != n_expected:
            raise ValueError(f'Catalog count {index} differs from Stage 3 geometry count {n_expected}; reindex upstream.')
        vec.flush(); del vec
        vec_tmp.replace(out/'crop_vectors.npy'); rows_tmp.replace(out/'catalog.jsonl')
        write_json(out/'frame_bindings.json', bindings)
        used = ('crop_vectors.npy', 'catalog.jsonl', 'frame_bindings.json')
        done = {'config': config, 'count': index, 'frames': frame_count, 'dimension': dim,
                'files': {n: {'bytes': (out/n).stat().st_size, 'sha256': sha(out/n)} for n in used},
                'counts': dict(counts), 'note': 'No matching, object assignments, or PLY fusion yet.'}
        write_json(out/'catalog_complete.json', done)
        print(f'Catalog prepared: {index:,} observations, {frame_count} keep frames; no CUDA/model loaded.', flush=True)
        return done
    except BaseException:
        if 'vec' in locals():
            del vec
        raise


class Catalog:
    def __init__(self, out, verify=True):
        self.out = Path(out).resolve(); self.done = read_json(self.out/'catalog_complete.json')
        self.source = self.done['config']['source']
        # Cheap current source IDs first; don't accidentally combine a new QA selection with old groups.
        if sha(Path(self.source['features'])/'run.json') != self.source['feature_run_sha256']:
            raise ValueError('Feature run changed since catalog preparation.')
        if sha(Path(self.source['features'])/'summary.json') != self.source['feature_summary_sha256']:
            raise ValueError('Feature selection changed. Prepare a new catalog.')
        if sha(Path(self.source['dataset'])/'plan.json') != self.source['plan_file_sha256']:
            raise ValueError('Capture plan changed.')
        if digest(read_json(Path(self.source['segmentation'])/'quality_policy.json')) != self.source['quality_policy_hash']:
            raise ValueError('Stage 2 quality policy changed.')
        if verify:
            for name, entry in self.done['files'].items():
                checked_file(self.out/name, entry)
        self.rows = list(lines(self.out/'catalog.jsonl'))
        self.vectors = np.load(self.out/'crop_vectors.npy', mmap_mode='r', allow_pickle=False)
        if len(self.rows) != self.done['count'] or self.vectors.shape != (len(self.rows), self.done['dimension']):
            raise ValueError('Catalog array/row count mismatch')
        if any(r['index'] != i or r['frame_status'] != 'keep' or r['mask_status'] != 'keep'
               or not r['eligible_geometry'] for i, r in enumerate(self.rows)):
            raise ValueError('Catalog violates strict keep-only ordering/selection')
        self.bindings = read_json(self.out/'frame_bindings.json')
        self.centers = np.array([r['center'] for r in self.rows], dtype=np.float32)
        self.lo = np.array([r['lo'] for r in self.rows], dtype=np.float32)
        self.hi = np.array([r['hi'] for r in self.rows], dtype=np.float32)
        station_names = sorted({r['station_id'] for r in self.rows})
        station_idx = {name: i for i, name in enumerate(station_names)}
        self.stations = np.array([station_idx[r['station_id']] for r in self.rows], np.int32)
        self.lookup = {(r['frame_id'], r['mask_index']): i for i, r in enumerate(self.rows)}
        self.id = self.done['config']['catalog_id']

    def cosine(self, q, chunk=8192):
        query = norm_rows(self.vectors[q]); scores = np.empty(len(self.rows), np.float32)
        for start in range(0, len(scores), chunk):
            v = norm_rows(self.vectors[start:start+chunk])
            scores[start:start+len(v)] = v @ query
        return np.clip(scores, -1, 1)

    def candidates(self, q, global_k=24, local_k=24, bbox_padding=.30):
        scores = self.cosine(q); allowed = self.stations != self.stations[q]
        sources = {}; selected = []
        def choose(order, budget, source):
            per_station = Counter(); taken = 0
            for i in order:
                i = int(i)
                if not allowed[i] or per_station[int(self.stations[i])] >= 2:
                    continue
                per_station[int(self.stations[i])] += 1; taken += 1
                if i not in sources:
                    selected.append(i); sources[i] = []
                sources[i].append(source)
                if taken >= budget:
                    break
        if global_k:
            choose(np.argsort(-scores, kind='stable'), global_k, 'global_appearance')
        # Spatial hypothesis proposals use bounding *regions*, not identical centroids.
        gap = np.maximum(np.maximum(self.lo - self.hi[q], self.lo[q] - self.hi), 0)
        distances = np.linalg.norm(gap, axis=1)
        local = np.flatnonzero(allowed & (distances <= bbox_padding))
        if local_k and len(local):
            choose(local[np.argsort(-scores[local], kind='stable')], local_k, 'local_geometry')
        return [(i, float(scores[i]), sources[i]) for i in selected]

    def seed_indices(self, count=24):
        # Broad/flat regions are not good diagnostic seeds, but remain candidate observations.
        size = np.linalg.norm(self.hi-self.lo, axis=1)
        eligible = [i for i, r in enumerate(self.rows)
                    if r['area_px'] >= 256 and r['area_fraction'] <= .6 and size[i] <= 3.5
                    and not any(h in r['hints'] for h in ('possible_floor_like_surface', 'possible_high_horizontal_surface'))]
        if not eligible:
            raise ValueError('No suitable automatic seeds. Use inspect with a known frame/mask; do not include review masks.')
        def quality(i):
            r = self.rows[i]
            return np.log1p(r['area_px']) * r['geometry_fraction'] * min(1., r['crop_effective_patches']/4)
        ordered = sorted(eligible, key=lambda i: (-quality(i), self.rows[i]['observation_id']))
        result = []; stations = set()
        for min_separation in (.40, .15, 0.):
            for i in ordered:
                if i in result or self.stations[i] in stations:
                    continue
                if result and np.min(np.linalg.norm(self.centers[result]-self.centers[i], axis=1)) < min_separation:
                    continue
                result.append(i); stations.add(int(self.stations[i]))
                if len(result) >= count:
                    return result
        return result
