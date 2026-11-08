# SITESENSE 3D — read-only asset and coordinate inspection

This is a separate diagnostic tool. It does **not** edit `service_demo/`,
`asset_bindings.json`, workbook values, map files, original scene geometry, model
weights, CUDA settings, or previously exported objects. It does not run segmentation
or retrain anything.

## Purpose

Distinguish three different problems:

1. **Identity binding:** a synthetic workbook ID is bound to the wrong visual region.
2. **Position/geometry:** the saved target differs from the current region, or the
   region includes incorrect surfaces / its box center is not a visible part.
3. **Coordinate or camera transform:** scale, floor offset, axes, or a saved camera
   matrix disagree between stages.

A geometrically plausible coordinate cannot establish physical asset identity.
The workbook explicitly contains placeholder scene anchors. Alternative coordinate
modes in this tool are diagnostic hypotheses, **not automatic corrections**.

## Install

Close the dashboard so its asset bindings cannot change during the audit. Save
`sitesense_asset_inspector.zip` into Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/sitesense_asset_inspector.zip -d .
source .venv/bin/activate
```

The ZIP adds only `asset_inspection/`. Audit/export use existing NumPy, SciPy,
Pillow and PyYAML. Reading copied original Gaussian centers uses existing plyfile.
Desktop 3D viewing optionally uses Open3D; the HTML/PNG reports work without it.
Do not upgrade any working packages for this step.

## 1. Fast audit of all assets

```bash
bash asset_inspection/run.sh audit \
  --bundle outputs/service_demo/main_v1 \
  --out outputs/asset_inspection/audit_v1
```

Open:

```bash
xdg-open outputs/asset_inspection/audit_v1/index.html
```

This is a static, offline report, not the web demonstration and not a new server.
It lists the exact saved asset->region bindings and shows their map positions.
It also plots the workbook's ORIGINAL placeholder coordinates separately.

The audit reads scale/floor from your prepared dashboard and compares them with
the region scene and enhanced-scene manifests. It verifies the map-YAML hash and
independent world/grid conversion. No GPU is needed.

## 2. Extract existing bound objects and visual evidence

Use a NEW output directory, since outputs are deliberately not overwritten:

```bash
bash asset_inspection/run.sh inspect \
  --bundle outputs/service_demo/main_v1 \
  --out outputs/asset_inspection/objects_v1 \
  --assets all \
  --environment outputs/ros/valmet_colored_cloud.npz \
  --views 3 \
  --copy-gaussians
```

To inspect only particular problematic assets, replace `--assets all` with, for
example, `--assets ASSET-001,ASSET-013` using IDs in your audit CSV.

The environment NPZ must ALREADY be in the existing map frame. The earlier 200k
file is a good diagnostic context. Use `valmet_colored_cloud_full.npz` for denser
context. Omit `--environment` to inspect objects without a room context cloud.

Open:

```bash
xdg-open outputs/asset_inspection/objects_v1/index.html
```

For each asset, follow **Inspect**. It shows:

- The workbook class and the exact currently bound region ID; the name is not
  used as proof that the region is that equipment.
- The retained region cloud in XY and XZ views.
- Stored target (red), current bounding-box center (gold), and an actual observed
  surface point nearest the median (teal).
- Original image/mask evidence and reprojected geometry.
- Per-camera transform/depth checks and explicit alternatives to compare.

**What is extracted?** `bound_object_map.ply` is the cleaned existing region
assigned by the current binding. If an asset is only a manual map pin and has no
region, the tool does NOT invent an object. It can export a nearby spatial cloud
crop, clearly called `context_*`, not a semantic segmentation.

`--copy-gaussians` copies each available `gaussians_original.ply` unchanged into
the diagnostic folder. It does not transform only its XYZ or rerun extraction.
Original Gaussian coordinate frame is retained. The corresponding map-frame
Gaussian centers are exported separately for ordinary point display.

### Outputs

```text
outputs/asset_inspection/objects_v1/
  index.html
  audit.json
  summary.json
  assets.csv
  assets_current_map.png
  workbook_placeholder_map.png
  ASSET-001/
    index.html
    asset.json
    bound_object_map.ply
    bound_object_map.npz
    bound_object_gaussians_original.ply   # optional original-frame copy
    gaussian_centers_map.ply/.npz         # when available
    context_binding.ply/.npz              # spatial context, not an object
    context_region_center.ply/.npz
    context_region_surface.ply/.npz
    context_workbook_raw.ply/.npz
    context_workbook_as_scene.ply/.npz
    context_binding_as_scene.ply/.npz
    context_undo_double_transform.ply/.npz
    context_swap_xy.ply/.npz
    cloud_XY.png
    cloud_XZ.png
    v00_crop.png
    v00_projection.jpg
    v00_checks.json
    ...
```

The context crop is a cube with half-width 1.5 provisional metres by default.
Change with `--radius 2.0`. It is capped at 100,000 points per crop by default;
change `--max-context`. Full existing bound objects are not downsampled by that cap.

## 3. Optional Open3D desktop inspection

```bash
bash asset_inspection/run.sh view \
  --out outputs/asset_inspection/objects_v1 \
  --asset ASSET-001 \
  --mode binding
```

This shows the bound cloud, subdued local environment, target markers, and axes.
Open3D's usual drag/orbit, wheel/zoom and pan controls apply. An OpenGL desktop is
needed. If Open3D is unavailable, use the exported PLYs in an existing point-cloud
viewer or the supplied static report; no new install is necessary for audit/export.

The mode changes the compared target and spatial context only. It **never moves
or transforms the bound object** to make an incorrect candidate look aligned.

### Hypotheses you can inspect

| `--mode` | Meaning |
|---|---|
| `binding` | Actual stored map XYZ. Start here; no conversion. |
| `region_center` | Current box center; tests for stale target after cleanup. |
| `region_surface` | Actual retained point near the median; tests a hollow/occluded center aim. |
| `workbook_raw` | Synthetic anchor interpreted as map XYZ; NOT registered. |
| `workbook_as_scene` | Synthetic anchor treated as original Gaussian XYZ; NOT justified registration. |
| `binding_as_scene` | Convert saved XYZ to map; only relevant if it was incorrectly saved in raw GS axes. |
| `undo_double_transform` | Undo one transform; only relevant if it was accidentally applied twice. |
| `swap_xy` | Axis-swap diagnostic. This is a reflection, NOT an approved rigid transform. |

Do not choose a transformation merely because its point happens to lie near some
geometry. Prefer agreement with known source masks/calibration and real asset
identification evidence.

## 4. Optional original-scene GPU check

The existing source images already provide a faster diagnostic. To additionally
render a selected source viewpoint using the current dashboard transform:

```bash
bash asset_inspection/run.sh render \
  --out outputs/asset_inspection/objects_v1 \
  --asset ASSET-001 \
  --mode binding \
  --view-index 0
```

To keep that source camera position but aim it toward a different target hypothesis:

```bash
bash asset_inspection/run.sh render \
  --out outputs/asset_inspection/objects_v1 \
  --asset ASSET-001 \
  --mode region_surface \
  --view-index 0 \
  --aim-at-target
```

The original whole-room Gaussian PLY is loaded once for each command. It uses the
existing loader and gsplat with `backgrounds=None`. It saves raw RGB/depth/alpha,
actual camera matrices and a separate target-cross overlay; it does not paint over
raw RGB or run target recognition. Close other GPU renderers first.

A projection cross at the correct pixel can still be occluded or lie inside an
object. The report records target depth versus visible expected depth to help
interpret this, not to certify visibility. Camera axes and camera-vs-world
inverses are kept explicit.

## Interpretation

- **Correct cloud + correct source projections + wrong equipment name:** fix the
  asset-to-region binding later; do not flip axes globally.
- **Stored target differs from current cleaned region:** a stale binding, intentional
  off-center target, or wrong source region is possible. Inspect the difference.
- **Manual pin Z=1.0:** this came from the map-click default. Horizontal position may
  be fine while pitch is wrong for an object at a different height.
- **Camera scale/floor/pose disagree:** genuine transform/metadata inconsistency
  needs investigation. The audit flags it before any live changes are made.
- **Correct target but final view obstructed:** this may be approach/visibility,
  rather than a coordinate error. A box center is not always a visible surface.
- **Target falls in an occupied cell:** equipment is an obstacle; this alone is NOT
  a mapping error. Navigation should stop at a separate reachable approach point.

## Checks and limits

The tool hashes the prepared config, bindings, workbook snapshot, map files and
scene manifest before/after to detect concurrent changes. It writes only into a
new separate report directory and refuses to overwrite an existing nonempty one.
No returned suggestion is applied to the dashboard.

Focused synthetic smoke checks covered known/stale targets, manual pin height,
map and camera inverses, mask/depth reprojection, exports and monitored source-file
preservation. These are NOT Valmet-scene registration accuracy tests. Real model
inference, GPU rendering, actual object identity and the user's current local
bindings have not been validated here.

## Useful files to share

`summary.json`, `assets.csv`, and one problematic asset's `asset.json` plus
`v00_projection.jpg`. No need to upload the entire PLY or all 30 object clouds.
