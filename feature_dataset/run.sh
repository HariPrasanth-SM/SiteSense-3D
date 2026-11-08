#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing project .venv/bin/python. Use the existing working environment." >&2
  exit 1
fi
# No CUDA path changes, model downloads, package upgrades, or ROS sourcing.
exec "$ROOT/.venv/bin/python" -m feature_dataset "$@"
