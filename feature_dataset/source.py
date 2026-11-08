"""Read the actual Stage 2 contract; never reinterpret review masks as kept."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .common import sha, digest, read_json, write_json, write_jsonl


class Source:
    def __init__(self, dataset, segmentation):
        from segment_dataset.storage import load_plan
        self.dataset = Path(dataset).expanduser().resolve()
        self.seg = Path(segmentation).expanduser().resolve()
        self.plan = load_plan(self.dataset)
        self.seg_run = read_json(self.seg/'run.json')
        self.source = read_json(self.seg/'source.json')
        self.summary = read_json(self.seg/'summary.json')
        self.policy = self.summary['policy_hash']
        if self.plan['plan_hash'] != self.seg_run['plan_hash'] or self.plan['plan_hash'] != self.source['plan_hash']:
            raise ValueError('Capture and segmentation plan hashes do not match.')
        if self.seg_run['run_id'] != self.summary['run_id']:
            raise ValueError('Stage 2 run/summary identity mismatch.')
        if self.source['plan_file_sha256'] != sha(self.dataset/'plan.json'):
            raise ValueError('Capture plan bytes differ from Stage 2 source binding.')
        if digest(read_json(self.seg/'quality_policy.json')) != self.policy:
            raise ValueError('Stage 2 quality policy changed without reselection. Finish reselect first.')
        self.frames = self.plan['frames']
        self.by_id = {r['frame_id']: r for r in self.frames}
        self.binding = {'schema': 'valmet.features.source.v1', 'dataset': str(self.dataset),
                        'segmentation': str(self.seg), 'plan_hash': self.plan['plan_hash'],
                        'segmentation_run_id': self.seg_run['run_id']}

    def read_rows(self, fid):
        p = self.seg/'analysis'/self.policy/fid/'observations.jsonl'
        with p.open() as f: rows = [json.loads(l) for l in f if l.strip()]
        for i, row in enumerate(rows):
            expected = f"{self.seg_run['run_id']}:{fid}:m{i:04d}"
            if row['observation_id'] != expected or row['mask_index'] != i or row['frame_id'] != fid:
                raise ValueError(f'Observation ordering/ID mismatch in {fid}.')
            if row['policy_hash'] != self.policy or row['run_id'] != self.seg_run['run_id']:
                raise ValueError(f'Observation policy/run mismatch in {fid}.')
            if row['decision']['status'] not in ('keep', 'review', 'exclude'):
                raise ValueError('Unsupported decision status')
        return rows

    def inspect(self, view, verify=True):
        from segment_dataset.storage import archive_valid
        fid = view['frame_id']; folder = self.dataset/'frames'/fid
        raw = self.seg/'frames'/fid
        audit = read_json(self.seg/'audit'/f'{fid}.json')
        if audit['policy_hash'] != self.policy: raise ValueError(f'Stale frame audit: {fid}')
        record = dict(view, frame_status=audit['status'], policy_hash=self.policy)
        if not (raw/'complete.json').exists():
            if audit['status'] != 'exclude':
                raise ValueError(f'{fid}: incomplete Stage 2 segmentation. Finish Stage 2 first.')
            return dict(record, segmented=False, mask_count=0, skip_reason='no_segmentation_archive')
        source_stamp = sha(folder/'complete.json')
        if not archive_valid(raw, self.seg_run['run_id'], source_stamp, verify=verify):
            raise ValueError(f'{fid}: corrupt or incompatible Stage 2 mask archive.')
        stamp = read_json(folder/'complete.json')
        if stamp['plan_hash'] != self.plan['plan_hash'] or stamp['frame_id'] != fid:
            raise ValueError('Capture completion identity mismatch')
        # Only input RGB and calibration need rereading here. Raw depth was checked
        # by Stage 2; Stage 3 references but does not consume depth for visual features.
        for name in ('rgb.png', 'camera.json'):
            info = stamp['files'][name]; p = folder/name
            if p.stat().st_size != info['bytes'] or (verify and sha(p) != info['sha256']):
                raise ValueError(f'Source checksum mismatch: {p}')
        inf = read_json(raw/'inference.json'); rows = self.read_rows(fid)
        with np.load(raw/'masks.npz', allow_pickle=False) as a:
            shape = [int(x) for x in a['shape']]
            if a['bits'].shape != (shape[0], (shape[1]*shape[2]+7)//8) or str(a['bitorder']) != 'little':
                raise ValueError('Invalid bit-packed masks')
        if shape[0] != len(rows) or shape[0] != inf['mask_count']:
            raise ValueError(f'Mask/metadata count mismatch in {fid}')
        cam = read_json(folder/'camera.json')
        if shape[1:] != [cam['height'], cam['width']]: raise ValueError('Mask/camera resolution mismatch')
        key = {'plan_hash': self.plan['plan_hash'], 'segmentation_run_id': self.seg_run['run_id'],
               'frame_id': fid, 'source_complete_sha256': source_stamp,
               'mask_complete_sha256': sha(raw/'complete.json')}
        status = Counter(r['decision']['status'] for r in rows)
        strict = [r for r in rows if r['decision']['status']=='keep' and audit['status']=='keep']
        return dict(record, segmented=True, mask_count=len(rows), shape=shape,
                    input_hash=digest(key), input_binding=key, mask_status_counts=dict(status),
                    strict_keep_masks=len(strict), strict_geometry_masks=sum(r['decision']['eligible_geometry'] for r in strict))

    def load_pixels(self, record):
        from segment_dataset.storage import unpack_masks
        fid = record['frame_id']
        with Image.open(self.dataset/'frames'/fid/'rgb.png') as im: rgb = np.array(im.convert('RGB'))
        masks, scores, boxes = unpack_masks(self.seg/'frames'/fid/'masks.npz')
        if masks.shape != tuple(record['shape']) or masks.shape[1:] != rgb.shape[:2]:
            raise ValueError('Pixel/mask size mismatch. Never resize source masks to conceal an error.')
        return rgb, masks


def audit(source, out):
    from .common import json_lines
    base = out/'source.json'
    if base.exists() and read_json(base) != source.binding:
        raise ValueError('Different source at this feature output. Use a new directory.')
    records = []
    for i, view in enumerate(source.frames, 1):
        records.append(source.inspect(view))
        if i%100 == 0 or i == len(source.frames): print(f'Input audit {i}/{len(source.frames)}', flush=True)
    write_json(base, source.binding)
    write_jsonl(out/'inventory.jsonl', records)
    counts = Counter()
    for r in records:
        counts['frames'] += 1; counts['segmented_frames'] += r['segmented']
        counts['masks'] += r['mask_count']; counts['strict_keep_masks'] += r.get('strict_keep_masks', 0)
        counts['strict_geometry_masks'] += r.get('strict_geometry_masks', 0)
        for k, v in r.get('mask_status_counts', {}).items(): counts['mask_'+k] += v
    summary = dict(counts=counts, policy_hash=source.policy, source=source.binding,
                   default_selection='mask keep AND frame keep; review/exclude archived only',
                   depth_rechecked=False, note='RGB/calibration/masks integrity checked; no model inference yet.')
    write_json(out/'audit_summary.json', summary)
    write_json(out/'policies'/f'{source.policy}.json', read_json(source.seg/'quality_policy.json'))
    print(json.dumps(summary['counts'], indent=2), flush=True)
    return records


def pilot(records, count=32):
    records = [r for r in records if r['segmented'] and r['mask_count']]
    count = min(count, len(records)); chosen = {}; groups = defaultdict(list)
    for r in records:
        if r['frame_status']=='keep': groups[r['station_id']].append(r)
    # Prefer two different headings/heights per station; reserve up to 1/4 for QA cases.
    keep_budget = max(1, count - min(count//4, sum(r['frame_status']!='keep' for r in records)))
    names = sorted(groups); station_n = min(len(names), max(1, (keep_budget+1)//2))
    if names:
        for j in np.linspace(0, len(names)-1, station_n, dtype=int):
            group = groups[names[j]]
            for k in np.linspace(0, len(group)-1, min(2, len(group)), dtype=int):
                if len(chosen)<keep_budget: chosen[group[k]['frame_id']] = group[k]
    others = [r for r in records if r['frame_status']!='keep']
    if others:
        for j in np.linspace(0, len(others)-1, min(count-len(chosen), len(others)), dtype=int):
            chosen[others[j]['frame_id']] = others[j]
    remaining = [r for r in records if r['frame_id'] not in chosen]
    if remaining and len(chosen)<count:
        for j in np.linspace(0, len(remaining)-1, min(count-len(chosen), len(remaining)), dtype=int):
            chosen[remaining[j]['frame_id']] = remaining[j]
    return list(chosen.values())
