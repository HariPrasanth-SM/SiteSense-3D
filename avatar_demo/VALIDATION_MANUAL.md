# Manual-controls update — validation

## Executed

- `python -m compileall -q avatar_demo`: passed.
- `xvfb-run -a python -m unittest discover -s avatar_demo/tests -v`: **49 passed**.
  - 14 original core tests (map import, planning, movement, transformations).
  - 22 new CPU manual tests (forward/reverse/turn, unicycle curve, per-cell segment
    checks, no wall tunneling or corner cuts, unknown handling, unrestricted
    motion, strict versus point-centre mode, pitch and input-source management).
  - 13 Tk UI tests (holds/releases, X11 repeat handling, stop latch, entry focus,
    window focus loss, mouse buttons, plan cancellation, guard switches, bookmarks,
    pitch during an automatic walk, adjustable speed, exact camera-pose metadata).
- A 1440x900 desktop preview with an artificial map and gradient RGB/depth
  placeholders was rendered and inspected. All interactive controls were inside
  the window; no input buttons/fields were clipped. This is not a rendered Valmet
  environment and is not evidence of camera/GPU performance.
- Original `core.py`, `render_worker.py`, `run.sh`, `requirements.txt`, and package
  entry points were compared byte-for-byte with `valmet_avatar_demo.zip`: unchanged.

## Not tested / limitations

- No available NVIDIA GPU was used to execute the CUDA worker in this environment.
- The user's actual Gaussian PLY and map PGM/YAML are not available locally. The
  visible screenshots are not enough to establish real collision geometry.
- The pre-existing map's body-height assumptions and scale remain provisional.
- Noclip is unrestricted planar visual inspection, not a collision-free human path.
- Point-centre mode does not certify footprint or head clearance, nor does it
  identify small objects that a person could safely step over.
- Tests ran on this environment's Python/Tk; the Ubuntu 22.04/Python 3.10 laptop
  still needs its own short manual-input and camera check. The code uses Python
  3.10-compatible syntax and keeps the previously working renderer unchanged.

## Suggested local sequence

1. Back up the old `avatar_demo` folder and apply the update.
2. Test `--no-render`, arrow keys, W/S and Stop.
3. Enable live camera with the existing environment; begin with `--render-hz 2`.
4. Inspect one uncertain region with map checks OFF and a red warning visible.
5. Save a bookmark after the camera catches up; keep current and captured pose
   differences in the review record.
