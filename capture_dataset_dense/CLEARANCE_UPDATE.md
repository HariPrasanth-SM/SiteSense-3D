# Valmet dense capture — 25 cm camera spawning buffer (v2.1.0)

This update replaces only the `capture_dataset_dense/` tool. It does not change
`capture_dataset/`, `avatar_demo/`, the renderer, the CUDA environment, the source
occupancy map, the Gaussian PLY, or previously saved camera plans and images.

## Behavior

- Default `--clearance 0.25` (25 cm under the existing provisional scale).
- `--wall-clearance` is an alias for `--clearance`.
- All spawned camera stations — automatic, manual pins, and corridor samples —
  must pass the requested buffer from occupied cells and the map exterior.
- Automatic sampling is also buffered away from unknown cells.
- Explicitly approved pins/corridors can be in or near gray cells, but must still
  satisfy the occupied-boundary/exterior buffer. Unknown is not labelled a wall.
- New manual pins that are too close are refused with an explanation.
- Imported near-boundary pins and interpolated corridor samples are skipped and
  logged, NEVER moved to a different corridor or silently accepted.
- The remaining station budget is filled from eligible known-free cells. If fewer
  stations fit, the tool reports that and still distributes the requested 2,000
  frames among the retained stations. It never silently reduces the buffer.
- A corridor that is too narrow may have few or no retained camera stations.
  Review its centerline and map instead of assuming it can support the buffer.
- Height range 0.5–2.5 m, heading/pitch sampling, seed 42 and metadata format remain
  as before. Only the spawn selection and its provenance have changed.

This is NOT a physical collision guarantee. It measures distance from the existing
2-D occupied boundary, which already includes inflation. It does not remove that
inflation. Scale 1.43 is provisional, not measured. Overhead pipes, low clutter,
inaccurate maps, missing geometry and poor splat appearance can still spoil views,
especially at the 0.5 m and 2.5 m camera-height extremes.

## Install

Stop the capture selector and rendering process before updating. Back up the tool:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
cp -a capture_dataset_dense "capture_dataset_dense_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/valmet_capture_clearance_update.zip -d .
```

Use `-o` so updated code replaces the previous dense-tool code. No pip install,
GPU-driver change, or modification to `.bashrc` is needed.

## Reuse your existing corridor selection, with new camera positions

Use a NEW output folder. Camera changes cannot be applied by resuming an old plan:

```bash
bash capture_dataset_dense/run.sh select \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --selection outputs/datasets/valmet_views_2000_v1/selection.json \
  --out outputs/datasets/valmet_views_2000_clearance25_v2 \
  --positions 200 \
  --frames 2000 \
  --height-min 0.5 \
  --height-max 2.5 \
  --spacing 0.25 \
  --corridor-spacing 0.25 \
  --clearance 0.25 \
  --seed 42
```

Use the actual location of your previous `selection.json`. To start with a blank
selection, omit `--selection`. To produce the new plan without a selection window,
replace `select` with `plan`; the same selection and parameters apply.

The saved selection preserves your ORIGINAL pins and corridor line. Skipped
samples are not erased from that record, so the requested line can be reviewed or
resampled with an explicitly revised buffer later.

## Preview

The selector now shows:

- **Red tint:** occupied-boundary/exterior clearance exclusion. No spawned station
  is allowed here, even a gray-approved pin or corridor sample.
- **Amber tint:** known-free cells too close to unknown for automatic sampling.
  Explicit reviewed pins/corridors may use them if the wall/exterior buffer passes.
- **Red X:** skipped imported pin or interpolated corridor sample (NOT a camera).
- **Mouse hover:** conservative occupied-boundary clearance for that grid cell.
- **Status:** required buffer and number of skipped explicit samples.

Click **SAVE PLAN + CLOSE**, then inspect:

```bash
xdg-open outputs/datasets/valmet_views_2000_clearance25_v2/camera_positions.png
```

The terminal prints the minimum retained clearance and reports any reduced count.
`spawn_clearance_report.json` records rejected sample coordinates and reasons.
`camera_positions.csv` records the clearance of each retained camera station.
`frames.jsonl` and `frame_poses.csv` include the station's clearance with every view.

## Render the pilot, then the complete batch

```bash
bash capture_dataset_dense/run.sh render \
  --out outputs/datasets/valmet_views_2000_clearance25_v2 \
  --pilot

xdg-open outputs/datasets/valmet_views_2000_clearance25_v2/qa/dense.html
```

After checking the pilot:

```bash
bash capture_dataset_dense/run.sh render \
  --out outputs/datasets/valmet_views_2000_clearance25_v2
```

The existing renderer loads the Gaussian scene once; completed frames in this NEW
plan are resumable. Existing datasets remain available, but do not inherit the new
clearance rule automatically. Do not edit their plan files or mix old frames into
the new folder: poses and plan hashes are different.

## Adjusting the distance

- `--clearance 0.20`: at least 20 cm map-boundary buffer.
- `--clearance 0.25`: default 25 cm.
- `--clearance 0.30`: at least 30 cm, usually fewer eligible locations.

Choose a fresh output folder for every changed setting. The minimum is enforced
conservatively at grid-cell centers, so some retained distances will exceed the
requested number. A 0.50 m-wide passage is not guaranteed to fit two 0.25 m margins
on a discrete, already-inflated map.

## Implementation detail

The tool uses SciPy's Euclidean distance transform in map units. EDT measures
cell-center distances. The code subtracts half a cell diagonal, so its reported
clearance is a conservative lower bound to the entire blocked-cell square, not
only to that cell's center. Padding treats the map exterior as blocked.

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

No attempt is made to infer physical walls from the Gaussian PLY in this patch.
Automatic frames still need rendered-view review; a 2-D buffer cannot guarantee
sharp images or height-dependent 3-D clearance.

## Validation and rollback

57 dense-tool tests passed (including 20 new clearance tests); all 30 original
capture tests also passed. Tests covered occupied-cell edges/corners, map exterior,
automatic positions, manual and corridor filtering, explicit gray permissions,
count/frame preservation, metadata, source-map preservation and the Tk selector.
A synthetic-map UI screenshot was checked at 1440x900. No CUDA or Valmet-scene
rendering was performed for this patch.

Run the tests in your existing environment if needed:

```bash
source .venv/bin/activate
python -m unittest discover -s capture_dataset_dense/tests -v
```

Tk tests require a desktop display. Tests do not need the proprietary PLY or GPU.

For rollback, close the tool, move this folder aside, and copy back the backup
folder created during installation. Do not remove your `outputs/` directory.
