#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Expected your existing .venv/bin/python in $ROOT. No packages were installed." >&2
  exit 1
fi
# CPU geometry; cap BLAS threads by default to avoid memory contention on the laptop.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${VALMET_BLAS_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$PY" -m associate_dataset "$@"
