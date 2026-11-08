from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def checked_file(path, entry):
    path = Path(path)
    if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
        raise ValueError(f'Input integrity mismatch: {path}')


def norm_rows(a):
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or np.any(norms <= 1e-8):
        raise ValueError('Non-finite or zero eligible descriptor')
    return a / norms


def disjoint(out, *sources):
    out = Path(out).expanduser().resolve()
    for src in sources:
        src = Path(src).expanduser().resolve()
        if src == out or src in out.parents or out in src.parents:
            raise ValueError(f'Output must be separate from source: {src}')
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def lock(out):
    import fcntl
    with (Path(out) / '.stage4.lock').open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another Stage 4 command is writing this output.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def module_hash():
    p = Path(__file__).parent
    return digest({f.name: sha(f) for f in sorted(p.glob('*.py'))})


def export_ply(path, xyz, rgb):
    """Write binary little-endian XYZRGB without Open3D dependency."""
    path = Path(path); xyz = np.asarray(xyz, dtype=np.float32); rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
        raise ValueError('Invalid point-cloud arrays')
    data = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, axis in enumerate(('x', 'y', 'z')):
        data[axis] = xyz[:, i]
    for i, c in enumerate(('red', 'green', 'blue')):
        data[c] = rgb[:, i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment diagnostic expected-depth samples in provisional map metres\n'
              f'element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as f:
        f.write(header.encode('ascii')); f.write(data.tobytes())
