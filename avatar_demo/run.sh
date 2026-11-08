#!/usr/bin/env bash
# Execute with bash avatar_demo/run.sh from anywhere; this does not change .bashrc.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
    echo "Expected your existing gsplat environment at $ROOT/.venv" >&2
    exit 1
fi
source "$ROOT/.venv/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
if [[ -d "$CUDA_HOME/bin" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MAX_JOBS="${MAX_JOBS:-2}"
exec python -m avatar_demo "$@"
