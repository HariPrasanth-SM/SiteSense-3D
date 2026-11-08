from __future__ import annotations
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from .common import digest, read_json, write_json, write_npz, export_ply, sha, module_hash
from .geometry import GeometryStore, check_pair


def _save_surface(folder, surface, prefix):
    write_npz(folder/f'{prefix}.npz', map_xyz_m=surface.xyz, rgb=surface.rgb, source_uv=surface.uv)
    export_ply(folder/f'{prefix}.ply', surface.xyz, surface.rgb)


def run_queries(catalog, query_indices, policy, out, global_k=24, local_k=24):
    """Independent star-shaped hypotheses. No connected-component instance clustering."""
    out = Path(out)
    recipe = {'schema': 'valmet.stage4.verify.v1', 'catalog_id': catalog.id,
              'policy': asdict(policy), 'global_candidates': global_k, 'local_candidates': local_k,
              'query_indices': list(map(int, query_indices)), 'code_sha256': module_hash()}
    recipe['run_id'] = digest(recipe)
    path = out/'run.json'
    if path.exists() and read_json(path) != recipe:
        raise ValueError('This pilot output has different queries/settings/code. Use a new --name.')
    write_json(path, recipe)
    store = GeometryStore(catalog, policy)
    results = []; started = time.perf_counter()
    for qid, q in enumerate(query_indices):
        q = int(q); row = catalog.rows[q]; name = f'q{qid:03d}_{row["frame_id"]}_m{row["mask_index"]:04d}'
        folder = out/'queries'/name; folder.mkdir(parents=True, exist_ok=True)
        done_path = folder/'result.json'
        if done_path.exists():
            record = read_json(done_path)
            if record['run_id'] != recipe['run_id']:
                raise ValueError('Old query result does not match run')
            # Recheck all source files actually used in the query, including raw depth arrays.
            for fid in record['input_bindings']:
                store.frame(fid)
            for fn, entry in record.get('files', {}).items():
                p = folder/fn
                if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
                    raise ValueError(f'Corrupt query output: {p}; remove this query folder to recompute.')
            results.append(record); print(f'[{qid+1}/{len(query_indices)}] reused {name}', flush=True)
            continue
        t0 = time.perf_counter(); store.used_bindings = {}
        seed = store.surface(q)
        _save_surface(folder, seed, 'seed_surface_map')
        candidates = catalog.candidates(q, global_k, local_k, policy.bounds_gate_m)
        edges = []; projection_arrays = {}
        for i, (c, appearance, sources) in enumerate(candidates):
            edge, debug = check_pair(catalog, store, q, c, appearance, sources)
            edge['edge_id'] = digest({'run': recipe['run_id'], 'q': row['observation_id'],
                                      'c': catalog.rows[c]['observation_id']})
            if debug is not None:
                key = f'e{i:03d}'
                for direction, dd in zip(('ab', 'ba'), debug):
                    projection_arrays[f'{key}_{direction}_uv'] = dd['uv']
                    projection_arrays[f'{key}_{direction}_code'] = dd['code']
                edge['projection_debug_key'] = key
            edges.append(edge)
        if projection_arrays:
            write_npz(folder/'projection_checks.npz', **projection_arrays)
        # Reject no pair by transitive chaining. A group is only directly supported by its seed.
        supported = [e for e in edges if e['status'] == 'supported']
        def strength(e):
            return min(e['query_to_candidate']['support_fraction'], e['candidate_to_query']['support_fraction']) * e['crop_cosine']
        supported.sort(key=lambda e: (-strength(e), e['candidate_index']))
        representative = []; used_stations = {row['station_id']}
        for e in supported:
            r = catalog.rows[e['candidate_index']]
            if r['station_id'] not in used_stations:
                representative.append(e); used_stations.add(r['station_id'])
            if len(representative) >= 8:
                break
        xyz = [seed.xyz]; rgb = [seed.rgb]
        sample_observation_indices = [np.full(len(seed.xyz), q, np.int32)]
        for e in representative:
            s = store.surface(e['candidate_index'])
            xyz.append(s.xyz); rgb.append(s.rgb)
            sample_observation_indices.append(np.full(len(s.xyz), s.index, np.int32))
        pts = np.concatenate(xyz); colors = np.concatenate(rgb); owner = np.concatenate(sample_observation_indices)
        # Diagnostic ONLY: no exclusive assignment, no voxel carving, no measured surface claim.
        write_npz(folder/'supported_surface_preview_map.npz', map_xyz_m=pts, rgb=colors,
                  catalog_indices=owner)
        export_ply(folder/'supported_surface_preview_map.ply', pts, colors)
        lo, hi = np.percentile(pts, [2, 98], axis=0)
        distinct_positions=[np.array(row['camera_position'],float)]
        for e in representative:
            position=np.array(catalog.rows[e['candidate_index']]['camera_position'],float)
            if min(np.linalg.norm(position-p) for p in distinct_positions)>=.10:
                distinct_positions.append(position)
        hypothesis = {'hypothesis_id': f'{recipe["run_id"]}:{name}', 'seed_observation_id': row['observation_id'],
                      'all_directly_supported_observation_ids': [e['candidate_observation_id'] for e in supported],
                      'preview_member_observation_ids': [row['observation_id']] + [e['candidate_observation_id'] for e in representative],
                      'status': 'tentative_seed_region' if len(distinct_positions) >= 3 else 'insufficient_multiview_support',
                      'distinct_preview_stations': len(used_stations),
                      'distinct_preview_positions_10cm': len(distinct_positions),
                      'partial_surface_center_map_m': np.median(pts, axis=0).tolist(),
                      'partial_surface_bounds_p02_p98_m': [lo.tolist(), hi.tolist()],
                      'semantic_label': None, 'object_id': None,
                      'navigation_goal': None, 'complete_object_boundary': False,
                      'overlap_with_other_hypotheses_allowed': True,
                      'scope': 'Direct seed matches only; no transitive or scene-wide merge. PLY is an expected-depth sample union.'}
        write_json(folder/'hypothesis.json', hypothesis)
        # Include every frame used for retrieval surfaces; cached frames also contribute provenance.
        used_fids = {row['frame_id']}
        used_fids.update(catalog.rows[e['candidate_index']]['frame_id'] for e in edges if not e.get('geometric_gate_only'))
        filelist = ['seed_surface_map.npz', 'seed_surface_map.ply', 'supported_surface_preview_map.npz',
                    'supported_surface_preview_map.ply', 'hypothesis.json']
        if (folder/'projection_checks.npz').exists():
            filelist.append('projection_checks.npz')
        record = {'query_name': name, 'run_id': recipe['run_id'], 'query_index': q,
                  'query': row, 'edges': edges, 'counts': dict(Counter(e['status'] for e in edges)),
                  'seconds': time.perf_counter()-t0, 'hypothesis': hypothesis,
                  'input_bindings': {fid: catalog.bindings[fid] for fid in sorted(used_fids)},
                  'files': {fn: {'bytes': (folder/fn).stat().st_size, 'sha256': sha(folder/fn)} for fn in filelist}}
        write_json(done_path, record); results.append(record)
        print(f'[{qid+1}/{len(query_indices)}] {name} | {len(edges)} tested | {record["counts"]} | {record["seconds"]:.1f}s', flush=True)
    edges = [e for r in results for e in r['edges']]
    from .common import write_json as wj
    for filename, subset in [('pairs.jsonl', edges), ('supported_edges.jsonl', [e for e in edges if e['status']=='supported']),
                              ('partial_edges.jsonl', [e for e in edges if e['status']=='partial_overlap']),
                              ('seed_hypotheses.jsonl', [r['hypothesis'] for r in results])]:
        tmp = out/(filename+'.tmp')
        with tmp.open('w') as f:
            for e in subset:
                f.write(json.dumps(e, separators=(',', ':'), allow_nan=False)+'\n')
        tmp.replace(out/filename)
    summary = {'schema': 'valmet.stage4.pilot_summary.v1', 'run_id': recipe['run_id'],
               'catalog_observations': len(catalog.rows), 'query_count': len(results),
               'pair_count': len(edges), 'pair_status': dict(Counter(e['status'] for e in edges)),
               'tentative_seed_groups': sum(r['hypothesis']['status']=='tentative_seed_region' for r in results),
               'seconds_this_command': time.perf_counter()-started,
               'selection': 'keep frame AND keep mask AND Stage2 geometry eligible AND valid Stage3 crop',
               'notes': ['No model inference. No new semantic labels or confirmed instance IDs.',
                         'Supported means heuristic joint evidence, NOT ground-truth instance identity.',
                         'One-sided overlaps, occluded comparisons and ambiguous cases remain separate.',
                         'PLYs are diagnostic visible-surface samples, NOT original Gaussian subsets.',
                         'No full-scene clustering or navigation goals. Overlapping groups are allowed.']}
    write_json(out/'summary.json', summary)
    return results
