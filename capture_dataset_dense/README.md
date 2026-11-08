# Valmet dense capture: 2,000 calibrated views with variable height

**v2.1.0 update:** all camera spawns now enforce a default 25 cm map-boundary
buffer, including pins and corridor samples. Read [CLEARANCE_UPDATE.md](CLEARANCE_UPDATE.md)
for upgrade commands, new previews, and the unchanged 3-D/scale limitations.
Use a NEW output dataset; existing plans are immutable.

An additive tool for the already-working Valmet project. This creates a NEW
`capture_dataset_dense/` module. It reuses the existing `capture_dataset/`
renderer and `avatar_demo/` loader/camera transforms, without changing either.

## What changed

- Default: **200 horizontal stations and 2,000 frames**, normally 10 views per station.
- Per-station camera heights are stratified and jittered through **0.5–2.5 provisional m**.
- Headings cover a full turn with per-station random phase and within-sector jitter.
- Pitch is randomized within height-dependent limits; low cameras are roughly level,
  and high cameras look down instead of deliberately aiming at the ceiling.
- Random seed defaults to **42**. The exact generated poses are saved in the plan;
  rendering/resuming does not generate new random poses.
- Draw a capture-only corridor centerline: stations are inserted along it at a
  target spacing of **0.25 provisional m**, then snapped/deduplicated to grid cells.
- Corridor stations reserve part of the total budget. They are not added on top.
- The remaining stations are spread over known-free cells. No unmarked unknown
  region is included automatically.
- Pilot normally contains **48 images**: eight stations, six heights/views each,
  including corridor/gray stations and room stations.
- All original RGB/depth/alpha/intrinsic/pose/checksum records remain available.
- Adds height-aware contact sheets, `frame_poses.csv`, and `sampling_summary.json`.

No FastSAM, DINOv3, training, reconstruction, navigation change, ROS service,
CUDA installation or package upgrade is performed.

## Install

Close the capture-selection window and stop any active capture before installing.
Save `valmet_capture_dense_update.zip` into Downloads, then:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_capture_dense_update.zip -d .
```

This is a new folder, so `-n` is appropriate for first installation. Keep the
existing `avatar_demo/`, `capture_dataset/`, and working `.venv/` present. Do NOT
replace the original renderer or run a pip upgrade for this update.

Expected structure:

```text
valmet-metaverse/
  avatar_demo/                  # unchanged
  capture_dataset/              # unchanged; original captures remain resumable
  capture_dataset_dense/        # new tool
  data/Industrial Boiler Room.ply
  outputs/ros/valmet_map.yaml
  outputs/ros/valmet_map.pgm
  .venv/                       # existing working torch/gsplat environment
```

## 1. Open the dense selector

```bash
bash capture_dataset_dense/run.sh select \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --out outputs/datasets/valmet_views_2000_v1 \
  --positions 200 \
  --frames 2000 \
  --height-min 0.5 \
  --height-max 2.5 \
  --spacing 0.25 \
  --corridor-spacing 0.25 \
  --seed 42
```

To reuse previous manual pins/exclusion disks, append:

```bash
--selection outputs/datasets/valmet_views_v1/selection.json
```

The old selection is optional; omit the argument if it is not present. Original
automatic stations are re-sampled at the new density, rather than copied.
Old pins and the previous dataset are not edited.

### Draw the narrow upper passage

1. Select **Draw corridor**.
2. Enable **Allow GRAY for capture ONLY**.
3. Click along the center of the passage from one end toward the other. Add
   intermediate vertices at bends; do not draw a straight line through equipment.
4. Press **Finish corridor**. The preview inserts many closely spaced stations
   along this line, including gray cells you explicitly approved for capture.
5. Add other passages with the same procedure, or use **Pin cameras** for individual
   viewpoints. When happy, press **SAVE PLAN + CLOSE**.

Occupied map cells and the map exterior cannot be included by the corridor tool.
A segment crossing an occupied cell is rejected; undo and add bends or inspect
that region before planning a different centerline. A line touching a blocked
cell corner is also rejected. This is a conservative 2-D selection check, NOT a
proof of 3-D clearance and NOT a navigation route.

Controls:

| Control | Effect |
|---|---|
| Pin cameras + left click | Add one camera station; gray requires opt-in |
| Draw corridor + left click | Add a polyline vertex |
| Finish corridor | Commit that line and allocate corridor samples |
| Undo corridor point/line | Remove last draft vertex, or last finished corridor |
| Right click | Undo a draft vertex; otherwise remove a nearby manual pin |
| Shift + left click | Exclude a 0.4 m disk from sampling |
| Undo exclusion | Remove latest exclusion |
| Recompute | Recalculate the spatial spread |
| SAVE PLAN + CLOSE | Save immutable plan, selection and visual previews |

The occupancy image is read from YAML/PGM, not inferred from the screenshot.
Unknown permission is stored as capture-only provenance. The navigation map stays
unchanged, and these camera poses must not be treated as verified service goals.

### Counts and spacing

- With 200 stations and 2,000 frames, each station has 10 views.
- With `--positions 160 --frames 2000`, each station has 12 or 13 views.
- A 6 m drawn corridor typically yields about 25 sample positions before grid
  snapping/duplicate removal, all within the total station budget.
- If fewer stations fit the requested free-cell spacing/clearance, the tool reports
  the smaller count. It still distributes the requested frame budget among those
  actual positions. It does not silently invent positions inside occupied cells.
- If manually pinned/corridor positions exceed the budget, saving is refused.
  Increase `--positions` or `--corridor-spacing`, or shorten the selected lines.
- Default spawn clearance is 0.25 provisional m from occupied-map boundaries/exterior
  for ALL stations; automatic sampling is also buffered from unknown. This is not
  avatar/body or height-dependent 3-D clearance. See CLEARANCE_UPDATE.md.
- Too-close imported pins/corridor samples are skipped and logged, not relocated.
  New pin clicks are rejected immediately. No silent buffer relaxation occurs.
- Automatic spacing is enforced for auto stations; explicit pins can be closer.

## 2. Height and angle distribution

Each station covers the entire selected height range using one jittered sample
per equal-width height stratum. For 10 views through 0.5–2.5 m, strata are 0.2 m
wide. Height order is shuffled independently of the heading sectors.

Headings cover 360 degrees with random per-station phase and within-sector jitter.
Pitch limits depend on the actual sampled camera height:

| Height | Pitch distribution |
|---|---|
| Below 0.9 m | -5 to 0 degrees |
| 0.9 to below 1.5 m | -12 to +4 degrees |
| 1.5 to below 2.0 m | -20 to +2 degrees |
| 2.0 m and above | -40 to -10 degrees |

Negative pitch looks down; positive looks up. These limits refer to the optical
axis, not all pixels. Ceiling/floor can still appear in the 70-degree vertical
field of view. No automatic object-aiming or visibility optimization is done.
The full Cartesian product of height x heading x pitch is NOT rendered: ten
varied views per station keep the total at 2,000.

The map scale remains **1.43**, floor at Gaussian **Y=1.1**. Height is measured
from that assumed flat floor in provisional metres. At height 2.5 m, the camera
can intersect pipes/equipment even where its floor-map cell is free. At height
0.5 m, nearby clutter can obstruct it. Pilot review is necessary.

`--height` controls image resolution (pixels). Use `--height-min/--height-max`
for camera elevation. Do not change the scale without regenerating related maps.

## 3. Inspect positions and render a pilot

```bash
xdg-open outputs/datasets/valmet_views_2000_v1/camera_positions.png
```

A second `camera_positions_labels.png` includes every station label at a larger
resolution. `camera_positions.csv` lists exact locations and selection provenance.
The station numbers are identifiers, NOT walking order.

Close the avatar renderer/RViz/FastSAM first so the original Gaussian scene is not
loaded on the laptop GPU multiple times. Then:

```bash
bash capture_dataset_dense/run.sh render \
  --out outputs/datasets/valmet_views_2000_v1 \
  --pilot
```

For a one-image smoke test, append `--limit 1`. Repeating the same pilot command
resumes the remaining pilot views.

Review:

```bash
xdg-open outputs/datasets/valmet_views_2000_v1/qa/dense.html
```

The dense sheets show frame ID, camera height, yaw, pitch, and gray-capture status.
The pilot includes low/middle/high elevations rather than only convenient eye
height. Review high/low cameras and corridor samples for intersections, very
close geometry, incorrect orientation and poor coverage. Automatic warnings are
heuristics; they do not prove geometric correctness. Frames are never discarded
solely because they contain ceiling, floor or an occlusion.

If a station or height range is unsuitable, create a revised plan in a NEW
output folder. Do not edit `plan.json` after capture begins.

## 4. Render all 2,000 views and resume

```bash
bash capture_dataset_dense/run.sh render \
  --out outputs/datasets/valmet_views_2000_v1
```

The existing persistent Stage 1 renderer loads the scene once, renders one camera
at a time and saves each frame immediately. The 2,000-frame budget does not mean
all images are retained in GPU RAM. Exact runtime is not known for this scene;
use the pilot's printed render/write timings. Do not run parallel capture jobs
on the 8 GB GPU for this test.

Completed pilot frames are reused, with checksum verification. Ctrl+C preserves
completed frames. Re-run the identical command to finish. Pose randomness exists
only at plan creation, not during retries. A lock prevents two writers to one dataset.
Existing scene/code/version binding checks remain active.

To capture a bounded number of NEW images in a session:

```bash
bash capture_dataset_dense/run.sh render \
  --out outputs/datasets/valmet_views_2000_v1 --limit 100
```

### Storage

For 2,000 images at 640x480, the three raw float32 arrays (scene depth, metre
depth, alpha) alone use approximately **6.87 GiB**, calculated as
`2000 * 640 * 480 * 12` bytes. RGB/validity images, metadata and previews add more.
Reserve **at least 15 GiB free** for this capture, and additional space for the
subsequent mask/embedding/point-cloud stages. The renderer also checks free space.
It does not train models or create masks in this stage.

## Saved records and downstream compatibility

The original calibrated frame format is preserved:

```text
outputs/datasets/valmet_views_2000_v1/
  plan.json                         # exact pre-generated frames, config, seed, hash
  selection.json                    # pins, explicit corridor lines and exclusions
  sampling_summary.json             # station/view/height counts
  camera_positions.png
  camera_positions_labels.png
  camera_positions.csv
  frames.jsonl
  frame_poses.csv                    # height, yaw, pitch, XY, ID and provenance
  index.jsonl                       # original full frame index, joinable IDs
  quality.csv                       # original QA
  quality_dense.csv                 # QA plus heights/angles/corridor references
  render_binding.json
  render_environment.json
  frames/
    s000_v000/
      rgb.png
      depth_scene.npy
      depth_m.npy
      alpha.npy
      valid.png
      camera.json                   # exact K/c2w/w2c/c2map, depth convention etc.
      qa.json
      complete.json                 # written last; frame checksum record
  qa/
    index.html                      # original report retained
    dense.html                      # height-aware report; pilot views first
    dense_rgb_001.jpg
    dense_depth_001.jpg
    ...
```

Use `(plan_hash, frame_id)` as the observation key. New view IDs such as
`s025_v007` identify station 25, view index 7. Do NOT infer height or yaw from
that index; read `frame_poses.csv`, `frames.jsonl` or `camera.json`.

K and exact poses are written for EVERY frame, even if intrinsics are shared.
Depth is expected camera-axis Z. `depth_scene.npy` is original scene units;
`depth_m.npy` uses the assumed scene-to-map scale. The depth thumbnails are only
for display. Raw arrays are retained for later segmentation/back-projection.
Neither alpha nor expected depth is verified collision geometry.

The tool uses the existing v1 frame schema with a `capture_dataset_dense/2.1.0`
generator identifier and additional sampling metadata. Mask/embedding/object
stages should read the frame records, not parse fixed-yaw filenames. No masks,
DINOv3 features or multi-view associations are computed here.

## Reuse a dense selection headlessly

```bash
bash capture_dataset_dense/run.sh plan \
  --selection outputs/datasets/valmet_views_2000_v1/selection.json \
  --out outputs/datasets/valmet_views_2000_v2 \
  --positions 160 --frames 2000 \
  --height-min 0.5 --height-max 2.5 --seed 42
```

The scene/map defaults are the same as the select command. Supply --map/--scene
when using different files. Exact selections are tied to the original map identity.

Regenerate reports after an interrupted render without initializing CUDA:

```bash
bash capture_dataset_dense/run.sh report \
  --out outputs/datasets/valmet_views_2000_v1
```

## Validation / limits

57 dense-tool tests passed, including 20 new clearance tests, with Tk tests
under a virtual display. All 30 original capture CPU/UI tests also passed unchanged. Tested exact frame
budget, repeated-seed pose generation, pitch/height bounds, heading strata,
corridor opt-in, occupied-cell rejection, source preservation, pilot coverage,
calibration consistency, raw frame writing/checksums, and GUI selection/save.
The selector was visually inspected at 1440x900 using an artificial map.

Actual CUDA rendering against the Valmet scene was NOT run here: the scene/GPU
are on the user's laptop. The renderer/loader/camera-transform source files are
byte-identical to the existing supplied packages. Nothing here proves a camera
pose is inside collision-free 3-D space or that 2,000 views observe every asset.

## Existing work stays intact

This ZIP contains only `capture_dataset_dense/`. Your existing `capture_dataset/`,
`avatar_demo/`, maps, PLY, CUDA environment and `valmet_views_v1` remain intact.
Continue using the original launcher for original captures. Do not copy this
folder's files into the original capture package.
