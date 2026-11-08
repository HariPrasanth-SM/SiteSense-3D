#!/usr/bin/env python3
"""Copy an allowlisted source/documentation snapshot into a NEW GitHub staging folder.

No application code, input data, or Git history is changed. No git/network/model command
is run. Use the working virtual environment to capture its package-version inventory.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

MODULES = (
    'service_demo', 'avatar_demo', 'manual_labeler', 'capture_dataset',
    'capture_dataset_dense', 'camera_spawn_soft', 'segment_dataset',
    'feature_dataset', 'associate_dataset', 'semantic_scene',
    'semantic_scene_plus', 'semantic_animation', 'semantic_showcase',
    'asset_inspection', 'scripts', 'ros_scripts', 'tools',
)
TEXT_SUFFIXES = {'.py', '.sh', '.js', '.html', '.css', '.md', '.txt', '.json',
                 '.yaml', '.yml', '.toml', '.svg', '.ini', '.cfg'}
EXAMPLE_SUFFIXES = {'.md', '.json', '.yaml', '.yml', '.pgm', '.png', '.jpg', '.jpeg', '.webp', '.csv'}
SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'node_modules',
             '.mypy_cache', '.ruff_cache', '.idea', '.vscode', 'build', 'dist'}
ROOT_FILES = ('README.md', 'RUN_DEMO.md', '.gitignore', 'requirements-demo.txt', 'LICENSE')
PACKAGES = ('numpy', 'scipy', 'Pillow', 'PyYAML', 'plyfile', 'torch', 'torchvision',
            'gsplat', 'ultralytics', 'opencv-python', 'open3d', 'open_clip_torch')


def permitted_path(p: Path, base: Path) -> bool:
    rel = p.relative_to(base)
    return not any(part in SKIP_DIRS or '_backup_' in part or '_saved_' in part
                   for part in rel.parts)


def redact_paths(value, project: Path):
    if isinstance(value, dict):
        return {k: redact_paths(v, project) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_paths(v, project) for v in value]
    if isinstance(value, str):
        prefix = str(project)
        if value == prefix or value.startswith(prefix + '/'):
            return '${PROJECT_ROOT}' + value[len(prefix):]
        # Earlier annotation files used this project root; redact only that known prefix.
        old = '/home/harry/SpaceBots_2026/valmet-metaverse'
        if value == old or value.startswith(old + '/'):
            return '${PROJECT_ROOT}' + value[len(old):]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path.cwd())
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kit', type=Path, default=Path(__file__).resolve().parents[1],
                   help='Documentation kit root; defaults to this script’s parent project.')
    p.add_argument('--include-company-data', action='store_true',
                   help='Explicitly copy the bundled workbook snapshot after permission review.')
    a = p.parse_args()
    project = a.project.expanduser().resolve()
    kit = a.kit.expanduser().resolve()
    out = a.out.expanduser().resolve()
    if not (project / 'service_demo').is_dir():
        p.error('The project must contain service_demo/. Run from your working project.')
    if out.exists() or out == project or project in out.parents or out in project.parents:
        p.error('Use a NEW, separate sibling directory, for example ../sitesense-submission.')
    out.mkdir(parents=True)
    copied, skipped = [], []

    def copy_file(src: Path, dest_rel: Path):
        if src.is_symlink() or not src.is_file():
            skipped.append(str(dest_rel) + ' (symlink/not a regular file)'); return
        if src.stat().st_size > 20 * 1024**2:
            skipped.append(str(dest_rel) + ' (>20 MiB; review separately)'); return
        dest = out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(dest_rel))

    for name in ROOT_FILES:
        src = (project / name) if name == 'LICENSE' else (kit / name)
        if src.is_file(): copy_file(src, Path(name))
    # Additional root Python helpers, but not arbitrary data/config files.
    for src in sorted(project.glob('*.py')):
        copy_file(src, Path(src.name))
    for name in MODULES:
        base = project / name
        if not base.is_dir(): continue
        for src in sorted(base.rglob('*')):
            if not src.is_file() or not permitted_path(src, base): continue
            if src.suffix.lower() not in TEXT_SUFFIXES: continue
            rel = src.relative_to(project)
            if name == 'service_demo' and 'data' in src.relative_to(base).parts:
                if src.name == 'workbook_snapshot.json' and a.include_company_data:
                    copy_file(src, rel)
                else:
                    skipped.append(str(rel) + ' (bundled data; not source)')
                continue
            if src.name.startswith('.env') or src.name.lower() in {'credentials.json', 'token.json'}:
                skipped.append(str(rel) + ' (credential-like name)'); continue
            copy_file(src, rel)
    for dirname in ('docs', 'examples', 'environment', 'tools'):
        base = kit / dirname
        if not base.is_dir(): continue
        allowed = TEXT_SUFFIXES | EXAMPLE_SUFFIXES | {'.gif'}
        for src in sorted(base.rglob('*')):
            if src.is_file() and permitted_path(src, base) and src.suffix.lower() in allowed:
                copy_file(src, Path(dirname) / src.relative_to(base))

    # Include the current map as a small inspection example; never edit its source YAML.
    map_yaml = project / 'outputs/ros/valmet_map.yaml'
    if map_yaml.is_file():
        try:
            import yaml
            meta = yaml.safe_load(map_yaml.read_text())
            image = Path(meta['image'])
            if not image.is_absolute(): image = map_yaml.parent / image
            if image.is_file():
                name = 'valmet_map' + image.suffix.lower()
                copy_file(image, Path('examples/map') / name)
                meta['image'] = name
                dest = out / 'examples/map/valmet_map.yaml'
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(yaml.safe_dump(meta, sort_keys=False))
            else: skipped.append('map image unavailable: ' + str(image))
        except (ImportError, ValueError, KeyError, OSError) as exc:
            skipped.append('map example not copied: ' + str(exc))

    ann = project / 'outputs/manual_labels/scene_v1/annotations.json'
    if ann.is_file():
        data = redact_paths(json.loads(ann.read_text()), project)
        data['publication_note'] = ('Inspection copy only: absolute project paths redacted. '
                                    'Use original runtime annotations and objects/ evidence for the demo.')
        dest = out / 'examples/manual_annotations.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2) + '\n')

    env = out / 'environment'; env.mkdir(exist_ok=True)
    versions = {}
    for name in PACKAGES:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    (env / 'versions.json').write_text(json.dumps(versions, indent=2) + '\n')
    (env / 'known_environment.txt').write_text(
        '# Inventory from the staging interpreter; not an automatic installation recipe.\n' +
        '\n'.join(f'{k}=={v}' for k, v in versions.items()) + '\n')
    try:
        result = subprocess.run(['git', '-C', str(project / 'vendor/dinov3'), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True)
        (env / 'dinov3_revision.txt').write_text(result.stdout.strip() + '\n')
    except (OSError, subprocess.CalledProcessError):
        skipped.append('DINOv3 Git revision unavailable; record the feature-run revision manually.')
    (out / 'STAGING_REPORT.json').write_text(json.dumps({
        'copied_source_and_documentation_files': len(copied),
        'included_modules': [m for m in MODULES if (out / m).is_dir()],
        'company_snapshot_included': a.include_company_data,
        'skipped': skipped,
        'notice': 'Review data permissions and staged files. This is not a runtime export or a Git push.'
    }, indent=2) + '\n')
    print(f'Staged {len(copied)} source/documentation files in {out}')
    print('Original project unchanged. Review STAGING_REPORT.json and add chosen query images.')
    print('Next: replace the Drive link and GIF placeholder, then initialize Git in this NEW folder.')

if __name__ == '__main__':
    main()
