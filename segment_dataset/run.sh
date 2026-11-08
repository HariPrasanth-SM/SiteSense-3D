#!/usr/bin/env bash
# Reuse current working environment; no ROS, CUDA path changes, downloads or installs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Expected existing Python at $ROOT/.venv/bin/python" >&2
  exit 1
fi
exec "$ROOT/.venv/bin/python" -m segment_dataset "$@"
