#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m camera_spawn_soft "$@"
else
  exec python3 -m camera_spawn_soft "$@"
fi
