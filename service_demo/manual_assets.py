"""Read-only manual-label -> workbook join. No spatial registration or inferred IDs.

The original service bundle is never modified. import-labels writes a NEW bundle.
All geometry comes from annotations; only non-spatial service records come from Excel.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import shutil
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from .common import read_json, atomic_json, sha

SCHEMA = 'sitesense.manual_asset_catalog.v1'


def normal_tag(value):
    text = unicodedata.normalize('NFKC', str(value)).strip().upper()
    text = text.translate(str.maketrans({'–':'-', '—':'-', '−':'-', '‑':'-'}))
    match = re.fullmatch(r'([A-Z]+)\s*-?\s*(\d+)', text)
    if not match:
        return text
    return f'{match.group(1)}-{int(match.group(2)):03d}'


def class_title(value):
    return str(value).replace('_', ' ').title()


def vector(value, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f'{name} must be three finite numbers.')
    if positive and np.any(a <= 0):
        raise ValueError(f'{name} must be positive.')
    return a


def safe_media_path(root, relative):
    if not relative:
        return None
    base = Path(root).expanduser().resolve()
    rel = Path(str(relative))
    if rel.is_absolute():
        raise ValueError('Annotation evidence paths must be relative to --labels-root.')
    path = (base / rel).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Annotation evidence path escapes --labels-root.')
    return path


def join_annotations(db, cfg, annotations):
    """Return filtered workbook, exact-coordinate bindings, catalog and audit report."""
    if annotations.get('schema') != 'sitesense.manual_labels.v1':
        raise ValueError('Expected manual-labeler annotations.json (sitesense.manual_labels.v1).')
    source = annotations.get('source', {})
    if annotations.get('frame') != 'map' or source.get('coordinate_frame') != 'map':
        raise ValueError('Manual coordinates must already be in the map frame; no automatic conversion is performed.')
    for key in ('scale', 'floor_y'):
        if key not in source or not math.isclose(float(source[key]), float(cfg[key]), rel_tol=0, abs_tol=1e-7):
            raise ValueError(f'Annotation {key} does not match the working demo. No scale/offset correction will be applied.')
    tags = {}
    for aid, asset in db['assets'].items():
        tag = normal_tag(asset['master']['tag'])
        if tag in tags:
            raise ValueError(f'Duplicate workbook tag: {tag}')
        tags[tag] = aid
    active = [o for o in annotations.get('objects', []) if o.get('active', True)]
    if not active:
        raise ValueError('No active labelled objects.')
    bindings, catalog, matched, corrections, warnings = {}, {}, {}, [], []
    rows, seen_objects, seen_tags = [], set(), set()
    for obj in active:
        oid = str(obj['object_id']); original_label = str(obj['label']); tag = normal_tag(original_label)
        if oid in seen_objects or tag in seen_tags:
            raise ValueError(f'Duplicate active object ID or workbook tag: {oid} / {tag}. Not silently merged.')
        seen_objects.add(oid); seen_tags.add(tag)
        if tag not in tags:
            raise ValueError(f'{oid}: tag {original_label!r} has no exact/format-normalized workbook match. Correct the label rather than guess an asset.')
        aid = tags[tag]; record = copy.deepcopy(db['assets'][aid])
        actual_tag = record['master']['tag']; category = class_title(record['master']['asset_class'])
        xyz = vector(obj['target_map_xyz'], f'{oid}.target_map_xyz')
        box = obj.get('bbox_map', {})
        vector(box.get('center'), f'{oid}.bbox.center')
        vector(box.get('extent'), f'{oid}.bbox.extent', positive=True)
        if not math.isfinite(float(box.get('yaw_deg', 0))):
            raise ValueError(f'{oid}: invalid box yaw.')
        if obj.get('coordinate_frame', 'map') != 'map':
            raise ValueError(f'{oid}: object is not in map frame.')
        if obj.get('target_scene_xyz') is not None:
            g = vector(obj['target_scene_xyz'], f'{oid}.target_scene_xyz')
            s = float(cfg['scale']); f = float(cfg['floor_y'])
            check = np.array([s*g[0], s*g[2], s*(f-g[1])])
            if not np.allclose(xyz, check, rtol=0, atol=2e-5):
                raise ValueError(f'{oid}: map/scene target inconsistency. Geometry left unchanged.')
        if actual_tag != original_label:
            corrections.append({'object_id':oid,'field':'label','original':original_label,'display':actual_tag,'rule':'exact tag after typography/number-format normalization'})
        original_category = str(obj.get('category', ''))
        if original_category != category:
            corrections.append({'object_id':oid,'field':'category','original':original_category,'display':category,'rule':'canonical category for exact workbook tag; original retained'})
        item = copy.deepcopy(obj)
        item.update(asset_id=aid, label=actual_tag, category=category,
                    original_label=original_label, original_category=original_category,
                    workbook_asset_class=record['master']['asset_class'],
                    workbook_source=record['master'].get('_source'),
                    metadata_link='manual_user_tag_to_synthetic_workbook',
                    physical_asset_identity_verified=False)
        catalog[aid] = item
        record['manual_annotation'] = {'object_id':oid,'label':actual_tag,'category':category,
            'original_label':original_label,'original_category':original_category,'revision':obj.get('revision'),
            'semantic_source':'manual_user_label','metadata_link':'manual_user_tag_to_synthetic_workbook'}
        matched[aid] = record
        bindings[aid] = {
            'asset_id':aid,'status':'manual_scene_annotation','manual_object_id':oid,
            'region_id':None,'map_xyz':list(obj['target_map_xyz']),
            'bbox_map':copy.deepcopy(box),'reference_camera_pose':copy.deepcopy(obj.get('reference_camera_pose')),
            'anchor_original':copy.deepcopy(record.get('anchor_original')),
            'source':'annotations.json:objects/'+oid,
            'note':'Position and aim point from your manual scene annotation, used unchanged. Tag links to synthetic workbook service data; it does not verify the physical equipment identity. Workbook anchor coordinates are not used.'}
        rows.append({'object_id':oid,'tag':actual_tag,'category':category,'workbook_asset_id':aid,
                     'x':float(xyz[0]),'y':float(xyz[1]),'z':float(xyz[2]),
                     'annotation_revision':obj.get('revision'), 'workbook_source':record['master'].get('_source')})
    # Flag proximity without deduplication: separate labels and elevations stay separate.
    vals = list(catalog.values())
    for i, a in enumerate(vals):
        for b in vals[i+1:]:
            dist = float(np.linalg.norm(np.asarray(a['target_map_xyz']) - b['target_map_xyz']))
            if dist < .20:
                warnings.append({'kind':'nearby_annotation_targets','tags':[a['label'],b['label']],
                                 'distance_3d_m':dist,'action':'Both retained; map click chooser handles overlapping markers. No coordinates shifted.'})
    filtered = copy.deepcopy(db); filtered['assets'] = matched
    filtered['visible_asset_count'] = len(matched)
    filtered['notice'] = ('Scene locations and labels come from your manual annotations. Only those labelled objects are displayed. '
        'Service history, telemetry, condition and recommendations are synthetic workbook records joined by tag, not measured states of the pictured equipment. '
        'Original workbook anchors are retained as provenance only and never used for navigation.')
    report = {'schema':'sitesense.manual_asset_import.v1','active_objects':len(active),
              'matched_assets':len(matched),'hidden_unlabelled_workbook_assets':len(db['assets'])-len(matched),
              'mapping_rule':'exact normalized tag; no category-only or fuzzy identity assignment',
              'corrections':corrections,'warnings':warnings,'matches':rows,
              'coordinate_policy':'target_map_xyz and bbox_map copied exactly; no rescale, flips, XY offset or workbook-anchor coordinates',
              'categories':{c:sum(o['category']==c for o in catalog.values()) for c in sorted(set(o['category'] for o in catalog.values()))}}
    return filtered, {'revision':0,'assets':bindings}, {'schema':SCHEMA,'assets':catalog,'source':source}, report


def import_labels(args):
    base = Path(args.base).expanduser().resolve(); dest = Path(args.out).expanduser().resolve()
    src = Path(args.annotations).expanduser().resolve()
    root = Path(args.labels_root).expanduser().resolve() if args.labels_root else src.parent
    if dest == base:
        raise ValueError('Use a NEW --out bundle; the working bundle is preserved.')
    if dest.exists():
        raise ValueError(f'Output already exists: {dest}. Use a new directory; do not overwrite a live bundle.')
    cfg = read_json(base/'config.json'); db = read_json(base/'workbook.json'); annotations = read_json(src)
    filtered, bindings, catalog, report = join_annotations(db, cfg, annotations)
    # Verify the actual existing map and original Gaussian, not Excel coordinates.
    from avatar_demo.core import FloorMap
    m = FloorMap.load(cfg['map'])
    source = annotations['source']
    for name, path in [('map_yaml_sha256',Path(cfg['map'])), ('scene_sha256',Path(cfg['ply']))]:
        if source.get(name) and sha(path) != source[name]:
            raise ValueError(f'{name}: the annotations refer to a different input file. No automatic registration attempted.')
    import yaml
    map_meta = yaml.safe_load(Path(cfg['map']).read_text())
    image = Path(map_meta['image'])
    if not image.is_absolute(): image = Path(cfg['map']).parent/image
    if source.get('map_image_sha256') and sha(image) != source['map_image_sha256']:
        raise ValueError('The map image differs from the annotated map. Check paths before importing.')
    missing = []
    for aid, obj in catalog['assets'].items():
        cell = m.world_to_cell(obj['target_map_xyz'][:2])
        if not m.contains(cell):
            report['warnings'].append({'kind':'target_outside_map','tag':obj['label'],'xyz':obj['target_map_xyz']})
        # Occupied target is normal for a physical object: planner finds separate approach.
        for key in ('reference_crop','reference_camera','cloud_npz'):
            p = safe_media_path(root, obj.get(key))
            if p is None or not p.is_file(): missing.append({'tag':obj['label'],'object_id':obj['object_id'],'field':key,'path':str(p) if p else None})
        for obs in obj.get('observations',[]): safe_media_path(root,obs.get('path'))
    report.update(annotations_file=str(src), annotations_sha256=sha(src), labels_root=str(root),
                  missing_evidence=missing, created_at=time.time())
    # Prepare atomically; there is never a half-created destination bundle.
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=dest.name+'.tmp_', dir=dest.parent))
    try:
        new_cfg = copy.deepcopy(cfg)
        new_cfg.update(manual_assets='manual_assets.json', manual_annotations='manual_annotations.json',
                       manual_labels_root=str(root), manual_annotations_sha256=sha(src),
                       asset_source_mode='manual_annotations', origin_bundle=str(base),
                       travel_pitch_deg=0.0,
                       warning='Manual scene annotations joined to synthetic workbook records by tag. Existing map frame/scale retained; static 2D navigation is not physical clearance verification.')
        shutil.copy2(src,tmp/'manual_annotations.json')
        atomic_json(tmp/'config.json',new_cfg);atomic_json(tmp/'workbook.json',filtered)
        atomic_json(tmp/'asset_bindings.json',bindings);atomic_json(tmp/'manual_assets.json',catalog)
        atomic_json(tmp/'manual_import_report.json',report)
        with (tmp/'manual_asset_matches.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(report['matches'][0]));writer.writeheader();writer.writerows(report['matches'])
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print(f'IMPORTED {len(catalog["assets"])} labelled objects -> {dest}',flush=True)
    print('Categories:',report['categories'],flush=True)
    print('Corrected display spellings:',len(report['corrections']),flush=True)
    print('No spatial changes. Old synthetic asset positions excluded. Original bundle untouched.',flush=True)
    if missing:
        print(f'WARNING: {len(missing)} evidence file references not found. Navigation is available, but missing crops will not be replaced with unrelated images. Check --labels-root.',flush=True)
    print('Report:',dest/'manual_import_report.json',flush=True)


class ManualAssets:
    def __init__(self, bundle, cfg):
        path = Path(cfg['manual_assets'])
        if not path.is_absolute():path=Path(bundle)/path
        data=read_json(path)
        if data.get('schema') != SCHEMA: raise ValueError('Unexpected manual asset catalog schema.')
        self.items=data['assets']; self.source=data['source']; self.root=Path(cfg['manual_labels_root']).resolve()
        self.cache={}

    def info(self, aid):
        o=self.items[aid]
        p=safe_media_path(self.root,o.get('reference_crop'))
        available=p is not None and p.is_file()
        return {'object_id':o['object_id'],'label':o['label'],'category':o['category'],
                'original_label':o['original_label'],'original_category':o['original_category'],
                'target_map_xyz':o['target_map_xyz'],'bbox_map':o['bbox_map'],
                'reference_camera_pose':o.get('reference_camera_pose'),'revision':o.get('revision'),
                'crop_available':available,
                'crop_url':f'/manual-asset/{aid}/crop' if available else None,
                'focus_url':f'/manual-asset/{aid}/focus' if available else None,
                'geometry_note':o.get('geometry_note'),
                'metadata_link':o['metadata_link'],'physical_asset_identity_verified':False}

    def _observation_files(self, o):
        for row in o.get('observations',[]):
            op=safe_media_path(self.root,row['path'])
            folder=op.parent
            # Match the reference view rather than an arbitrary different observation.
            reference=safe_media_path(self.root,o.get('reference_crop'))
            if reference is not None and reference.parent!=folder: continue
            rgb=folder/'rgb.png';mask=folder/'mask_depth_supported.png'
            if not mask.exists():mask=folder/'mask_drawn.png'
            if rgb.is_file() and mask.is_file():return rgb,mask
        return None

    def media(self, aid, kind):
        if kind not in ('crop','focus','overlay'):raise ValueError('Unknown manual media kind.')
        key=(aid,kind)
        if key in self.cache:return self.cache[key]
        o=self.items[aid]
        # Both focused variants are tightly cropped. No full image in the target panel.
        pair=self._observation_files(o)
        if pair is not None:
            from .arrival import focused_image
            with Image.open(pair[0]) as im:rgb=np.array(im.convert('RGB'))
            with Image.open(pair[1]) as im:mask=np.array(im.convert('L'))>127
            if rgb.shape[:2]!=mask.shape:raise ValueError('Manual RGB/mask dimensions differ.')
            if mask.any():
                im=Image.fromarray(focused_image(rgb,mask,tint=(kind!='crop')))
            else:raise ValueError('Empty manual mask.')
        else:
            path=safe_media_path(self.root,o.get('reference_crop'))
            if path is None or not path.is_file():
                raise FileNotFoundError('Manual target crop missing. Keep the labeler objects/ folder and pass the correct --labels-root when importing.')
            with Image.open(path) as original:im=original.convert('RGBA')
            base=Image.new('RGBA',im.size,(238,234,241,255));base.alpha_composite(im);im=base.convert('RGB')
        im.thumbnail((1200,1000));out=io.BytesIO();im.save(out,'PNG')
        self.cache[key]=(out.getvalue(),'image/png');return self.cache[key]
