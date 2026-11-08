#!/usr/bin/env bash
# Uses the existing working environment. Does NOT edit .bashrc or LD_LIBRARY_PATH.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "Expected your existing gsplat Python environment at $ROOT/.venv" >&2
    exit 1
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ -d "$CUDA_HOME/bin" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export MAX_JOBS="${MAX_JOBS:-2}"
exec "$ROOT/.venv/bin/python" -m capture_dataset "$@"
