from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def json_lines(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        for row in rows: f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    tmp.replace(path)


def disjoint_output(out, *sources):
    out = Path(out).expanduser().resolve()
    for src in sources:
        src = Path(src).expanduser().resolve()
        if out == src or src in out.parents or out in src.parents:
            raise ValueError(f'Output must be separate from source: {src}')
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def lock(out):
    import fcntl
    with (Path(out) / '.features.lock').open('a+') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Another Stage 3 process is writing this folder.') from exc
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def complete(folder, run_id, input_hash):
    files = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in (folder/'features.npz', folder/'metadata.json')}
    if (folder/'dense.npz').exists():
        p = folder/'dense.npz'; files[p.name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    write_json(folder/'complete.json', dict(run_id=run_id, input_hash=input_hash, files=files))


def is_complete(folder, run_id, input_hash):
    try:
        rec = read_json(folder/'complete.json')
        if rec['run_id'] != run_id or rec['input_hash'] != input_hash: return False
        if not {'features.npz', 'metadata.json'}.issubset(rec['files']): return False
        for name, info in rec['files'].items():
            p = folder/name
            if Path(name).name != name or p.stat().st_size != info['bytes'] or sha(p) != info['sha256']: return False
        return True
    except (OSError, ValueError, KeyError, TypeError): return False


def source_hash(folder):
    return digest({str(p.relative_to(folder)): sha(p) for p in sorted(Path(folder).rglob('*.py'))
                   if 'tests' not in p.parts and '__pycache__' not in p.parts})
