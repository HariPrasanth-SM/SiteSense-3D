#!/usr/bin/env python3
"""Package existing object exports for inspection; never rerun extraction or edit data.

Writes a new ZIP with a manifest, cleaned clouds, original Gaussian subsets, and
selected supporting evidence. This is not a complete runnable dashboard archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

GEOMETRY = ('surface_clean.ply', 'surface_clean.npz', 'gaussians_original.ply',
            'gaussian_indices.npy', 'gaussian_centers_map.npz')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path.cwd())
    p.add_argument('--enhancement', type=Path, required=True)
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--environment', type=Path)
    p.add_argument('--zip', dest='archive', type=Path, required=True)
    a = p.parse_args()
    root = a.project.expanduser().resolve()
    def resolve(path):
        path = path.expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    enh, scene, archive = resolve(a.enhancement), resolve(a.regions), resolve(a.archive)
    if not (enh / 'regions').is_dir() or not (scene / 'scene.json').is_file():
        p.error('Expected enhancement/regions/ and semantic-scene/scene.json.')
    if archive.exists(): p.error('ZIP already exists. Choose a new filename to preserve previous exports.')
    for base in (enh, scene):
        if archive == base or base in archive.parents: p.error('Write the ZIP outside the input asset folders.')
    environment = resolve(a.environment) if a.environment else None
    if environment and not environment.is_file(): p.error(f'Environment cloud not found: {environment}')
    regions = [d for d in sorted((enh / 'regions').iterdir()) if d.is_dir() and
               any((d / name).is_file() for name in GEOMETRY)]
    if not regions: p.error('No prepared object export folders found.')
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_name(archive.name + '.partial')
    if tmp.exists(): p.error('A partial ZIP exists. Keep/remove it deliberately before retrying.')
    manifest = {'schema': 'sitesense.inspection_asset_bundle.v1',
                'region_count': len(regions), 'regions': [], 'files': {},
                'coordinate_notes': {
                    'surface_clean': 'map frame; provisional metres',
                    'gaussians_original': 'original Gaussian axes; attributes unchanged',
                    'environment': 'map-frame XYZRGB cloud when supplied'},
                'notes': ['Inspection exports, not verified distinct complete assets.',
                          'Regions may overlap. No new point ownership or label is assigned.',
                          'Metadata project paths are redacted; binary geometry remains unchanged.',
                          'This bundle is not an executable dashboard or pipeline-resume archive.']}
    if (enh / 'config.json').is_file():
        cfg = json.loads((enh / 'config.json').read_text())
        manifest['transform_metadata'] = {k: cfg[k] for k in ('scale', 'floor_y', 'transform') if k in cfg}

    def redacted(value):
        if isinstance(value, dict): return {k: redacted(v) for k, v in value.items()}
        if isinstance(value, list): return [redacted(v) for v in value]
        if isinstance(value, str):
            for prefix in (str(root), '/home/harry/SpaceBots_2026/valmet-metaverse'):
                if value == prefix or value.startswith(prefix + '/'):
                    return '${PROJECT_ROOT}' + value[len(prefix):]
        return value

    try:
        with zipfile.ZipFile(tmp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1,
                             allowZip64=True) as z:
            def add_file(src, dest):
                if src.is_symlink(): raise ValueError(f'Refusing symlink: {src}')
                digest = hashlib.sha256(); size = 0
                with src.open('rb') as inp, z.open(dest, 'w', force_zip64=True) as out:
                    while True:
                        chunk = inp.read(4 * 1024**2)
                        if not chunk: break
                        digest.update(chunk); size += len(chunk); out.write(chunk)
                manifest['files'][dest] = {'bytes': size, 'sha256': digest.hexdigest()}
            def add_json(src, dest):
                value = redacted(json.loads(src.read_text()))
                blob = (json.dumps(value, indent=2) + '\n').encode()
                z.writestr(dest, blob)
                manifest['files'][dest] = {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
                                          'metadata_paths_redacted': True}
            for name in ('scene.json', 'review.json', 'semantic_suggestions.json'):
                if (scene / name).is_file(): add_json(scene / name, 'metadata/' + name)
            if (enh / 'config.json').is_file(): add_json(enh / 'config.json', 'metadata/enhancement_config.json')
            if environment:
                add_file(environment, 'environment/' + environment.name)
                manifest['environment_file'] = 'environment/' + environment.name
            for i, folder in enumerate(regions, 1):
                rid = folder.name; entry = {'region_id': rid, 'files': []}
                for name in GEOMETRY:
                    src = folder / name
                    if src.is_file():
                        dest = f'objects/{rid}/{name}'; add_file(src, dest); entry['files'].append(dest)
                for name, base in [('region.json', scene / 'regions' / rid), ('enhancement.json', folder)]:
                    src = base / name
                    if src.is_file(): add_json(src, f'objects/{rid}/{name}')
                # Small representative evidence only; never copy all source frames/masks.
                for sub, base in [('views', scene / 'regions' / rid / 'views'),
                                  ('overlays', folder / 'views'), ('renders', folder / 'renders')]:
                    if base.is_dir():
                        images = sorted(q for q in base.iterdir() if q.is_file() and
                                        q.suffix.lower() in {'.png', '.jpg', '.jpeg'})[:9]
                        for src in images: add_file(src, f'objects/{rid}/{sub}/{src.name}')
                manifest['regions'].append(entry)
                print(f'[{i}/{len(regions)}] {rid}', flush=True)
            z.writestr('ASSET_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
            z.writestr('README.txt',
                'SiteSense — SpaceBots / Valmet challenge\n\n'
                f'{len(regions)} exported region folders. See ASSET_MANIFEST.json for exact contents.\n'
                'surface_clean.ply: ordinary colored point cloud in the map frame.\n'
                'gaussians_original.ply: original Gaussian attributes and axes; use a compatible splat renderer.\n'
                'Do not transform only Gaussian XYZ without accounting for covariance and appearance.\n'
                'These are incomplete, potentially overlapping region hypotheses, not verified asset identities.\n'
                'Metadata paths are redacted; binary geometry is unchanged.\n'
                'This inspection bundle is not the runtime bundle for the browser dashboard.\n'
                'Share only according to the source-scene provider permission.\n')
        tmp.replace(archive)
    except BaseException:
        print(f'Incomplete archive retained at {tmp}; source assets were not changed.')
        raise
    print(f'Saved {archive} ({archive.stat().st_size/1024**2:.1f} MiB), {len(regions)} region folders.')
    print('Upload the ZIP outside Git and replace GOOGLE_DRIVE_OBJECT_ASSETS_URL in README.md.')

if __name__ == '__main__':
    main()
