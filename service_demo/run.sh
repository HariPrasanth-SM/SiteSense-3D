#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Keep the already-working CUDA and Python setup intact. No ROS sourcing.
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="python3"; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$PY" -m service_demo "$@"
