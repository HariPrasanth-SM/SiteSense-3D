#!/usr/bin/env bash
# Additive launcher: no edits to ROS, CUDA libraries, existing launchers or source data.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "Expected your working gsplat environment at $ROOT/.venv" >&2
    exit 1
fi
if [[ ! -f "$ROOT/capture_dataset/common.py" || ! -f "$ROOT/avatar_demo/render_worker.py" ]]; then
    echo "Keep the previously installed capture_dataset/ and avatar_demo/ beside this folder." >&2
    exit 1
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ -d "$CUDA_HOME/bin" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export MAX_JOBS="${MAX_JOBS:-2}"
exec "$ROOT/.venv/bin/python" -m capture_dataset_dense "$@"
