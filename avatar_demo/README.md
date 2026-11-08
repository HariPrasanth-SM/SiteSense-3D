# Update: manual exploration controls

See **[MANUAL_CONTROLS.md](MANUAL_CONTROLS.md)** for installation, mouse/keyboard
controls, free-roam, head tilt, speed settings, and boundary bookmarks.

Defaults: 0.25 m/s, 60 deg/s; arrow keys drive/turn, W/S tilt, H levels view,
Space/Escape stop. Manual map checks default to input-free cells (green + amber),
with an optional stricter body-clearance checkbox. Automatic planning retains its
original strict footprint rules. The rendering worker and launcher are unchanged.

The original general navigation/rendering guide continues below, with the changed
key and reset behavior corrected.

---

# Valmet: standalone avatar navigation and Gaussian-camera demo

A local Python desktop prototype for the existing `valmet-metaverse` project.
No ROS nodes, RViz, Gazebo, web service or model training is required to run it.
It reads your existing ROS-format occupancy map as an ordinary file and renders
from the original Gaussian PLY in a persistent CUDA worker.

## What this version does

- Displays the floor map with a circular avatar and heading arrow.
- Chooses an initial position in a large connected, footprint-safe candidate free region.
- Accepts a manual start, a clicked target, typed map coordinates, or a JSON command file.
- In **Inspect object** mode, the target may be inside an occupied or unknown cell.
  A reachable **approach point** is selected nearby; the avatar never drives into
  the target simply because it was clicked.
- In **Go to free point** mode, a valid free destination is used (snapped to its cell center).
- Computes an 8-neighbor A* route, prohibits diagonal corner-cutting, and excludes unknown cells.
- Animates stop-turn-walk motion and stops at the endpoint. Object mode then turns to face the target.
- Renders forward-facing RGB and expected depth using `gsplat` and the actual current camera pose.
- Uses camera height **1.7 m**, adjustable in the UI. Positive pitch looks up; negative looks down.
- Saves plans, trajectory CSV, optional camera snapshots, and optionally every rendered frame.

This is kinematic avatar motion on a static grid, not wheel/leg physics, Nav2,
SLAM, or a real robot controller. The avatar is drawn on the floor map; it is not
inserted as a 3-D body into the Gaussian scene. The camera view is first person.

## Put the package in your existing project

Extract the provided archive **into the project root**. The result should be:

```text
~/SpaceBots_2026/valmet-metaverse/
├── .venv/                            # your existing working torch/gsplat environment
├── data/Industrial Boiler Room.ply   # original Gaussian PLY
├── outputs/ros/valmet_map.yaml
├── outputs/ros/valmet_map.pgm
├── scripts/                         # unchanged
├── ros_scripts/                     # unchanged
└── avatar_demo/
    ├── __init__.py
    ├── __main__.py
    ├── app.py
    ├── core.py
    ├── render_worker.py
    ├── run.sh
    ├── requirements.txt
    ├── README.md
    ├── VALIDATION.md
    └── tests/
```

Example when the archive is saved in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
# Existing users: close the app and back up before installing this update.
cp -a avatar_demo "avatar_demo_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/valmet_avatar_manual_controls_update.zip -d .
```

The archive contains code/documentation only. It does not contain Valmet's data.
The program never overwrites your occupancy map, PLY, ROS scripts, or earlier outputs.
Each launch creates a new timestamped directory under `outputs/avatar_demo/`.

## Dependencies — use the working rendering virtual environment

Stop RViz and previous GPU/rendering jobs for the initial test. You do not need to
source ROS in this terminal. Unlike the ROS publisher, this program runs in `.venv`.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
source .venv/bin/activate
python -m pip install -r avatar_demo/requirements.txt
python -c "import tkinter; print('Tkinter OK')"
```

If the last command reports a missing Tk module on Ubuntu 22.04:

```bash
sudo apt install python3-tk
```

Keep your installed PyTorch and gsplat versions. This package does **not** request
an upgrade or rebuild of them. `run.sh` activates `.venv`, selects CUDA 12.6 if no
CUDA_HOME is already set, and then starts the application. It leaves `.bashrc` unchanged.

## 1. Check the existing occupancy map

```bash
bash avatar_demo/run.sh --check-map
```

This prints the map dimensions, the number of free and footprint-safe cells, and
a candidate starting coordinate. It does not initialize CUDA or open a window.

The default map is `outputs/ros/valmet_map.yaml`. Its image path is interpreted
relative to that YAML. Only `mode: trinary` is supported here. The current
210 x 170 Valmet map and its resolution/origin are read from the file, not hardcoded.

## 2. Test navigation without the renderer first

```bash
bash avatar_demo/run.sh --no-render
```

In the window:

1. Choose **Click: set start**, then click a green cell. The start is reset to the
   center of that cell. This is explicitly a reset/teleport, not simulated travel.
2. Choose **Click: target**.
3. Choose **Inspect object** to click on equipment, or **Go to free point** for an
   ordinary free-space waypoint.
4. Click a target. The path is planned but motion does not begin yet.
5. Inspect the displayed route, then press **Walk / resume**.
6. Use **Pause**, Escape, or Space to stop. Use **Walk / resume** to resume an automatic route.
   Manual arrow-key/button movement clears the old plan; plan again after manual exploration.

The default auto-start is selected from the largest connected candidate region,
with high map clearance. It is **not automatically verified against the splat**.
Start clicks follow the selected manual collision policy. With checks OFF they
can reset to occupied/unknown cells for inspection, but automatic A* still needs
a green, strict-footprint start. Such resets are teleports, not navigated paths.

### Map colors

- Pale green: candidate free cells that pass the additional footprint check.
- Amber: free in the input map but excluded by the automatic footprint check;
  allowed in point-centre manual mode, blocked in strict manual mode.
- Dark gray: occupied in the input map.
- Gray: unknown in the input map.
- Blue circle/arrow: avatar and current orientation.
- Red cross: requested target (possibly inside an obstacle).
- Gold ring: selected reachable stopping point.
- Cyan ring: position used for the most recently displayed camera image.

### Target and approach point are different

Object mode uses a point target, a desired minimum stand-off of 0.8 m and a maximum
approach distance of 2.5 m. It selects only cells reachable from the current start.
The criterion prefers a stopping distance close to the requested stand-off, with
a small preference for a position near the start. It then runs A* to that cell.

These distances are from the target **point**, not a known asset surface. For a
large object, increase **Max approach** as needed. The map handles body clearance;
the stand-off controls the inspection distance. If no approach exists in the
specified band, the application refuses the mission rather than crossing obstacles.

Turning toward the target does **not** establish line of sight. A different object
or a wall can still obscure it. Confirm the final view in RGB. A later semantic
version can use an asset box and verified observation viewpoints.

## 3. Launch RGB and depth rendering

Close the map-only window and run:

```bash
bash avatar_demo/run.sh \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --camera-height 1.7 \
  --width 480 --height 360 \
  --render-hz 1
```

Use the **original Gaussian PLY**, not the exported 200k/1.2M point cloud. Rendering
uses its scales, rotations, opacity and spherical-harmonic appearance parameters.

The floor map and controls appear while the GPU worker loads the scene. The worker
loads it only once. It then renders the latest pose, with no unbounded pose backlog.
A stationary camera is not repeatedly re-rendered unless its settings change.

The render request rate is capped at 1 Hz by default. Actual rate depends on GPU,
viewpoint, scene size and renderer startup. No frame rate is guaranteed. The GUI
and motion loop are separate from the GPU process. The displayed image reports
its actual pose; the cyan map marker makes slow-frame lag visible.

The first GPU call may reuse/build the gsplat extension cache just as your existing
render script does. A visible GUI does not mean the first CUDA render has completed.

### Controls for first-person inspection

- **Camera height**: default 1.7 m above the assumed flat floor.
- **Pitch**: 0 means horizontal; e.g. -15 looks down at nearby equipment/floor.
- **Idle heading**: yaw in map coordinates, degrees; 0 faces map +X, 90 faces map +Y.
- Applying a manual heading pauses navigation; resuming turns back toward the next path segment.
- **Save camera frame** saves the last *actually rendered* frame with its matching camera pose.
- At arrival, object mode rotates in place to face the target and requests the updated camera image.

### Lower GPU-load fallback

```bash
bash avatar_demo/run.sh \
  --width 320 --height 240 --render-hz 0.5 --sh-degree 0
```

SH degree zero removes directional color detail but retains the original Gaussian
positions, scales and opacity. Lower resolution reduces image work; lowering the
request rate reduces average workload, not the memory cost of loading the scene.
Do not start simultaneous gsplat/FastSAM jobs while measuring this prototype.

If the renderer fails, its traceback is printed in the terminal and map navigation
remains usable. Reopen with `--no-render` for a clean CPU-only fallback.

## 4. Scale and camera coordinates

The map already uses the provisional scale from your existing export:

```text
map_x = 1.43 * gaussian_x
map_y = 1.43 * gaussian_z
map_z = 1.43 * (1.1 - gaussian_y)
```

The inverse for camera height h is:

```text
gaussian_x = map_x / 1.43
gaussian_y = 1.1 - h / 1.43
gaussian_z = map_y / 1.43
```

At h=1.7 m, Gaussian camera Y is approximately -0.0888. Camera yaw/pitch are
converted to the same Y-down Gaussian frame, with camera +X right, +Y down,
+Z forward. RGB/depth and saved poses share that frame.

Depth is gsplat's **expected camera-axis Z depth**, not Euclidean range and not a
measured depth sensor. The worker scales it by 1.43 for the provisional-metre
output, rejects nonfinite/nonpositive or low-alpha pixels, and retains alpha.
Alpha is rendering coverage, **not** a calibration of geometric confidence.

Changing `--scale` only in this program without regenerating the occupancy map
would misalign geometry. Keep 1.43 unless the map and associated asset coordinates
are regenerated consistently. Likewise, `--floor-y` assumes one flat floor.

## 5. Clearance and map limitations

Default avatar footprint radius: 0.25 m. Additional margin: 0.03 m. The planner
also adds half a grid-cell diagonal to be conservative about motion through
cells, not just their centers. Unknown cells and map boundaries block travel.

Your existing occupancy map already contains obstacle inflation. This package
preserves it and applies additional clearance; it does not attempt to reverse
that inflation. Therefore narrow passages can be rejected conservatively.

Do not make unknown cells free just to obtain a route. Review your map against
the scene, save a corrected map as a separate file, and pass that YAML with
`--map`. A visually plausible corridor is not automatically verified free space.

The model does not establish 1.7 m head clearance, floor continuity, stairs,
overhanging obstacles, robot dynamics or collision geometry from the splat.
A clear path in this grid can still be wrong in 3-D. Validate a short route using
RGB/depth before recording or presenting it. The software does not execute commands
on a physical robot.

## 6. Optional terminal commands / later semantic targets

Start the application with a command file:

```bash
bash avatar_demo/run.sh --command-file outputs/avatar_command.json
```

From another terminal, write a goal atomically. Replace the example coordinates
with your chosen map-frame target:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/avatar_command.json")
p.parent.mkdir(parents=True, exist_ok=True)
command = {"action": "goal", "x": 1.0, "y": 2.0, "mode": "object", "walk": True}
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(command))
tmp.replace(p)
PY
```

Supported commands:

```json
{"action": "goal", "x": 1.0, "y": 2.0, "mode": "object", "walk": true}
{"action": "goal", "x": 1.0, "y": 2.0, "mode": "point", "walk": false}
{"action": "start", "x": 0.0, "y": 0.0, "yaw_deg": 90}
{"action": "pause"}
```

The start command resets position and accepts only footprint-safe cells. The
example coordinates are illustrative, not claimed safe locations in your scene.
Future semantic detection can write the target X/Y through this interface without
rewriting the planner. Object elevation and 3-D line-of-sight planning are not
implemented; pitch can currently be set manually.

## 7. Saved outputs and recording

Every run creates:

```text
outputs/avatar_demo/<timestamp>/
├── settings.json
├── trajectory.csv
├── plan_001.json
├── snapshots/                  # when Save camera frame is pressed
│   ├── snapshot_001_rgb.png
│   ├── snapshot_001_depth.npy  # scene units, compatible with previous back-projection convention
│   ├── snapshot_001_depth_m.npy
│   ├── snapshot_001_alpha.npy
│   └── snapshot_001_camera.json
└── frames/                     # only with --record
```

To save every rendered camera frame:

```bash
bash avatar_demo/run.sh --record --render-hz 1
```

Recording saves images and raw arrays, not an encoded movie. It adds disk I/O and
can use substantial storage. An RGB/depth frame pair always carries the pose used
to render that pair, even when the avatar has since moved. `trajectory.csv` is the
UI's kinematic trajectory; match frames by their saved pose/time, not row number.

## 8. Tests

```bash
source .venv/bin/activate
python -m unittest discover -s avatar_demo/tests -v
```

Tests use artificial maps and do not access Valmet's data or CUDA. See
`VALIDATION.md` for exactly what was and was not tested before delivery.

## 9. Troubleshooting

| Symptom | Action |
|---|---|
| `No module named avatar_demo` | Run from the project root; confirm the extracted folder is `avatar_demo/` directly below it. |
| Missing `_tkinter` / `tkinter` | Install Ubuntu `python3-tk`; use your existing Python 3.10 venv. |
| `no display name` | Launch from the laptop's graphical desktop, not a headless SSH session. |
| No walkable start | Use `--check-map`; inspect unknowns and existing inflation. Review the map before changing footprint assumptions. |
| Click rejected | Start/exact goal is outside the clearance-safe region; use object mode for a target inside equipment. |
| No approach point | Increase Max approach for a large asset, select another target/start, or correct the map after inspection. |
| Slow startup | Check the terminal for gsplat initialization/compilation; the scene is loaded only once. |
| Wrong CUDA compiler | `run.sh` selects `/usr/local/cuda-12.6` unless CUDA_HOME was set already. Check `echo "$CUDA_HOME"` and `nvcc --version`. |
| CUDA memory error | Stop other GPU jobs; test the low-load command. Do not downgrade working torch/gsplat without inspecting the traceback. |
| Camera upside-down or misplaced | Check scale/floor settings and source scene; the code assumes your confirmed negative-Y-up convention. |
| Final object not visible | Facing a point does not verify visibility. Move the start/approach, adjust pitch, or choose a visible target for this prototype. |
| Camera appears behind current avatar | Images may be low FPS; cyan marker and image pose show the actual rendered position. Reduce speed for the demo. |

## Implementation references

- gsplat 1.5.3 rasterization conventions, RGB+ED and packed rendering:
  https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- Nav2 Humble map loader (YAML, trinary interpretation and image row order):
  https://github.com/ros-navigation/navigation2/blob/humble/nav2_map_server/src/map_io.cpp
- Python 3.10 Tkinter and multiprocessing:
  https://docs.python.org/3.10/library/tkinter.html
  https://docs.python.org/3.10/library/multiprocessing.html

The source code here is a new minimal implementation. It does not copy Nav2's map
loader source. No third-party scene, model weights, or company data is bundled.
