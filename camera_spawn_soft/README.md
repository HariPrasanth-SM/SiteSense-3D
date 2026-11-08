# Valmet — camera spawning with preferred, not mandatory, clearance

This is a separate **CPU-only camera-position tool**. It does not render images,
load the Gaussian PLY, run navigation, segment anything, or change existing maps,
`capture_dataset_dense`, `capture_dataset`, `avatar_demo`, or CUDA settings.

## Why this exists

The previous 25 cm rule discarded camera positions in narrow passages. Here:

- Occupied/dark cells are never camera locations.
- Green/free AND gray/unknown cells inside a **room outline you approve** are
  available for virtual captures. Gray is not converted to free in the source map.
- The room outline excludes exterior gray scan noise. Draw it around the room,
  including the upper gray corridor, not around individual green patches.
- There is **no hard minimum clearance**. A seed closer than the preferred 0.18 m
  is moved locally when a better non-occupied location is available. Otherwise a
  narrow position is retained and explicitly flagged.
- Relocation is at most 0.30 m of local grid-path length, never through dark cells,
  outside the chosen region, across an excluded disk, or diagonally through corners.
- Already-adequate seeds normally stay put. With insufficient room, choose the
  best locally achievable clearance, rather than snapping into another room.
- Unique locations are generated with spatial coverage, plus a desired gray-seed
  share (20%). Isolated gray specks are not specially oversampled, but remain
  eligible. No occupied specks are silently erased.
- A priority corridor line reserves samples before general room sampling.
- This is camera placement, NOT a path, safety claim, or human body model.

All distances refer to the **existing already-inflated occupancy map**, not the
physical wall. The script cannot distinguish walls from equipment using this
trinary map. Gray gaps are not asserted to be physically empty.

## Installation

Save `valmet_camera_spawn_soft.zip` in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_camera_spawn_soft.zip -d .
```

The archive adds `camera_spawn_soft/` only. It uses NumPy, SciPy, Pillow, PyYAML and
Tkinter already present in your working environment. No pip upgrade is requested.
The launcher uses `.venv/bin/python`, but does not change CUDA or library paths.

## Generate 100 or 200 positions

```bash
bash camera_spawn_soft/run.sh \
  --map outputs/ros/valmet_map.yaml \
  --out outputs/camera_spawns_soft_v1 \
  --positions 200 \
  --preferred-clearance 0.18 \
  --max-shift 0.30
```

Use `--positions 100` for fewer points, or 160 for an intermediate plan. Change
`--preferred-clearance` to 0.15 or 0.20 as desired. It is ALWAYS a preference.
Use a new output directory for a new plan; previous outputs are not overwritten.

### Window workflow

1. **Room outline**: left-click around the room's OUTER envelope, including the
   entire gray upper passage. Vertices may be on dark perimeter walls; interior
   dark cells remain blocked. Keep exterior gray noise outside the outline. Draw
   a simple, non-self-crossing polygon. Click **Finish shape**.
2. **Priority corridor** (recommended): click a few points along the gray passage
   centerline, then **Finish shape**. This allocates camera seeds approximately
   every 0.35 m. You do not need to click every spawn.
3. **Generate points**: inspect the station distribution, then **SAVE + CLOSE**.

`Pin camera` adds an individual preferred seed. A seed can shift locally during
sampling; original and final coordinates are both recorded.

Other controls:

- Right-click or **Undo**: remove a draft vertex, or the latest item of the current
  editing mode. Undo can also remove a recent exclusion disk.
- Shift+left-click: add a 0.30 m capture-exclusion disk. No map labels change.
- A new mode clears unfinished draft vertices but retains committed shapes.
- Point color **blue**: the preferred conservative map clearance was attained.
- Point color **orange**: less than preferred; intentionally KEPT.
- Purple ring: final location is gray/unknown in the original map.
- Thin seed-to-final lines show displacement, not navigable motion.
- Mouse hover gives world XY and the estimated map-clearance lower bound.

A corridor line is a coverage instruction, not a route. If part crosses a dark
cell, its seeds there are skipped and reported. Already non-dark narrow samples
are NOT dropped because of clearance. Follow the actual passage with bends.

If explicit pins/priority-corridor seeds exceed the total count, increase the count
or `--corridor-spacing` (e.g. 0.50), rather than silently discarding the priority.

### Reuse existing corridor selections

Append the actual path to the previous dense selector's selection file:

```bash
--selection outputs/datasets/valmet_views_2000_v1/selection.json
```

This imports original pins, corridor lines and exclusions. You still draw a room
outline once. Map hashes must match. An old selection preserves original requested
corridor points even if the old hard-clearance tool dropped their camera samples.

## Reuse the new outline without opening a window

```bash
bash camera_spawn_soft/run.sh \
  --map outputs/ros/valmet_map.yaml \
  --region outputs/camera_spawns_soft_v1/region.json \
  --out outputs/camera_spawns_soft_100_v2 \
  --positions 100 \
  --preferred-clearance 0.20 \
  --headless
```

Leave off `--headless` to reopen and adjust the saved outline/corridor selection.
Do not supply both `--selection` and `--region`.

## Output files

```
outputs/camera_spawns_soft_v1/
  camera_positions.png          # overview, no overlapping station labels
  camera_positions_labels.png   # larger labeled preview
  camera_positions.json         # exact original/final map XY, map state, warning flags
  camera_positions.csv          # same information in table form, plus preview GS XYZ
  region.json                   # approved outline, priority lines, seed pins, exclusions
  spawn_report.json             # counts, skipped occupied seeds, settings, caveats
  selection.json                # compatible pinned-position input for existing dense capture
  capture_allowed_mask.npy      # virtual capture permission; NOT a navigation map
```

Every station records its original/final position, local shift length, achieved
conservative clearance, whether it fell below the preference, and gray provenance.
The seed and settings are retained, so regeneration is reproducible.

If there are fewer eligible cells than requested positions, the actual count is
reported rather than manufacturing extra stations. No two stations use the same
cell. There is no strict station-spacing guarantee after the local adjustment;
minimum spacing is reported. This is deliberately a count/coverage-oriented tool.

## Coordinate scale and the 3 m room-height assumption

The existing ROS map already has resolution 0.0715 m/cell, from scale 1.43 applied
to 0.05 scene-unit cells. Keep the existing transformation:

```
x_map = 1.43 * x_GS
y_map = 1.43 * z_GS
z_map = 1.43 * (1.1 - y_GS)
```

Your approximate floor-to-overhead-structure height of 3 m is the earlier basis
for this provisional scale. This tool does NOT infer 3-D heights from a 2-D map,
set all walls to 3 m, or rescale the map to fit an image dimension.

`--camera-height 1.7` only records an illustrative Gaussian-space camera position
at each XY for later use. Actual per-frame heights, headings, intrinsics, and poses
are set/saved by the rendering stage. No images or orientations are created here.

### How clearance is calculated

SciPy's Euclidean distance transform measures distance to dark-cell centers. The
code subtracts half a grid-cell diagonal to report a conservative lower bound to
blocked cell squares. The map exterior also counts as blocked. Gray cells do not.
The approved outline limits motion/selection but is not claimed to be a physical
wall. No global erosion, opening, or forced inflation is used.

Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

## Later: use the finalized positions in the existing dense renderer

This is optional; first inspect the position preview. To make a new 2,000-frame
plan from exactly these camera positions:

```bash
bash capture_dataset_dense/run.sh plan \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --selection outputs/camera_spawns_soft_v1/selection.json \
  --out outputs/datasets/valmet_views_soft200_v1 \
  --positions 200 --frames 2000 \
  --height-min 0.5 --height-max 2.5 \
  --clearance 0 --seed 42
```

IMPORTANT: `--clearance 0` here means **do not apply the old tool's hard filter a
second time**. The preferred repositioning has already been performed. Final
narrow-camera exceptions are deliberately retained. Use `--positions` equal to the
number saved in `spawn_report.json` to avoid adding automatic stations at this handoff.

The old tool imports all these final stations as pins, so its corridor-count label
may read 0. The new `spawn_report.json` retains original corridor provenance.

The existing renderer then supports the normal pilot/full commands:

```bash
bash capture_dataset_dense/run.sh render --out outputs/datasets/valmet_views_soft200_v1 --pilot
```

None of these choices guarantee sharp views. Cameras at high or low elevations
may intersect geometry missing from the 2-D map. Later review the actual RGB/depth
and reduce extreme heights or exclude problematic stations; don't equate gray
capture permission or preferred clearance with physical free space.

## Tests and scope

```bash
source .venv/bin/activate
python -m unittest discover -s camera_spawn_soft/tests -v
```

27 tests passed in development (22 CPU + 5 Tk tests under a virtual display).
They cover exact counts/unique cells, a one-cell gray corridor, local inward moves,
no crossing walls or blocked corners, rotated map coordinates, source preservation,
GUI interaction, and serialization. A separate integration test generated a
2,000-frame plan using the existing dense tool and preserved all 100 test stations,
including below-preference gray stations.

The actual Valmet PGM/YAML and PLY are not present here, so no Valmet-scene planning
or GPU rendering was performed. All spatial tests used artificial maps.
