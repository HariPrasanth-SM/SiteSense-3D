# SITESENSE 3D — manual object annotation

A separate local browser application for the existing Valmet project. Explore with
joystick-style buttons, freeze the displayed RGB/depth frame, draw an object outline,
inspect/edit its 3D box, and save a manually named object with coordinate provenance.

**No workbook assets, procedural bindings, 4 m rescaling, axis-flip experiments,
DINO/FastSAM inference, reconstruction, or changes to the final dashboard are included.**
This tool labels what you actually see in the existing scanned environment.

## Install

Close the dashboard and other live Gaussian renderers first. Save the ZIP in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/sitesense_manual_labeler.zip -d .
source .venv/bin/activate
```

The archive adds `manual_labeler/` only. Keep `avatar_demo/`, including the earlier
manual-control update (`avatar_demo/manual.py`), present. It reuses its `FloorMap`,
`camera_matrices`, line/cell checks, and original Gaussian loader. There are no pip
installs, GPU driver changes, CUDA environment changes, external fonts, CDNs or APIs.
Use the existing `.venv` with numpy/scipy/Pillow/PyYAML/plyfile/torch/gsplat.

## Launch — one terminal

```bash
bash manual_labeler/run.sh serve \
  --bundle outputs/service_demo/main_v1 \
  --out outputs/manual_labels/scene_v1 \
  --width 640 --height 480 --render-hz 2 \
  --camera-height 1.7 \
  --speed 0.25 \
  --port 8772 \
  --open-browser
```

Browser: `http://127.0.0.1:8772`. Keep the terminal running. Stop with Ctrl+C.
The tool only READS the existing demo bundle's `config.json` for map/PLY paths,
scale and floor offset. It does NOT read/import/modify workbook asset bindings.
With the known existing config, the transform remains scale **1.43** and Gaussian
floor Y **1.1**. No 4 m recalibration or coordinate-range fitting is performed.

Without a prepared demo bundle, give paths explicitly:

```bash
bash manual_labeler/run.sh serve \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --out outputs/manual_labels/scene_v1 \
  --open-browser
```

If the usual bundle exists its transform is still read; otherwise the established
1.43/1.1 defaults apply. Do not silently change source geometry between sessions.
The output refuses a different source scene/map hash or scale/floor transform.

Initial view: highest-clearance start in the largest known-free component, unless
`--start X Y` is supplied. `--yaw 90` means facing map +Y; 0 faces map +X. These are
existing map coordinates, not spreadsheet coordinates.

## 1. Explore

- Hold **Forward/Backward** or Up/Down arrows to move along the heading.
- Hold **Left/Right** or Left/Right arrows to rotate (no strafing).
- **W/S** tilt the head up/down. **H** levels the view.
- **Escape/Space/Stop** stops movement.
- Eye height, pitch and speed are editable; click **Apply camera**.
- Keyboard controls are ignored while typing in an input/select field.
- Focus loss or missing command heartbeats stops motion (about 0.45 seconds maximum
  since the last received command). These are virtual controls, not a USB-gamepad driver.

The small floor map shows the avatar and manually annotated objects. Cyan indicates
the camera position of the displayed image. The image may lag current motion.

**Block dark cells** is on initially. It permits gray cells for inspection, blocks
occupied cells and the map exterior, and prevents diagonal corner-cutting. It is
point-center 2D checking, NOT a physical human-body or head-clearance guarantee.
Uncheck it for explicit free-roam inspection; a warning is shown. Neither choice
edits the map or establishes a navigable route.

For faster inspection, enable **Click to reposition** above the small map and click
another place. This is an explicit camera reset/teleport, not path finding.

## 2. Freeze the exact image before drawing

Stop at a clear view, let the image catch up, then press **Freeze / label** (F).
The exact displayed frame ID, RGB, depth, alpha, intrinsics and pose are pinned
server-side. Incoming live frames cannot replace this annotation image. Camera and
motion commands are blocked until **Resume live**. You can still adjust selection
and depth filters while frozen.

A freeze deliberately uses the image you saw, not a newer pose mislabeled with an
older image. Its position is shown at the top and in the saved camera record.
If a very stale frame is no longer cached, the tool asks for a fresh image rather
than silently switching to one.

## 3. Outline the object

Choose a tool below the RGB panel:

- **Draw rectangle:** drag a tight box around the object.
- **Draw polygon:** click vertices around its visible silhouette. Three or more
  vertices form a closed shape; Undo vertex removes the last point. Trace a simple
  perimeter (no self-intersections). This is often better for pipes/irregular shapes.
- **Click depth seed:** click an interior surface of the intended object. This
  enables a depth interval around that pixel (default +/-0.35 provisional metres).
- **Click aim point:** optionally pick a visible point to look at in the future demo.

Draw a new rectangle or clear the selection before changing from one object to another.
Click **Build / refresh 3D selection** after changing outline, seed or any filters.

### Exclude background

A rectangle alone contains background. To narrow it:

1. Use a polygon around the actual visible silhouette, or
2. Click a depth seed and adjust **Seed +/- m**.

The seed depth is **camera-axis Z**, not height above the floor and not distance along
the viewing ray. Near/Far fields also use camera-axis Z. A large tank viewed obliquely
may need a wider depth interval. Two objects at the same depth still require a better
outline; seed filtering is not semantic segmentation.

With **Keep seed-connected pixels only**, only the 4-connected selected component
containing the seed is used. Turn it off when an object has separated visible pieces.
The seed must have valid depth inside the outline; invalid seeds produce an error.

Default alpha support is >=0.5 and near/far is 0.15–25 m under the existing assumed
scale. Alpha is coverage, not calibrated geometric confidence.

## 4. Review the point cloud, box, and aim point

The small 3D panel displays up to 5,000 points from the current selection. Drag to
orbit, wheel to zoom, Reset to restore. The full retained sampled cloud is saved.
The gold box and red aim point are also projected over the frozen RGB image.

Up to 40,000 selected pixels are sampled deterministically for CPU processing;
the complete original outline and supported-pixel mask remain saved. Optional SOR
uses 24 neighbors and mean + 2 standard deviations to remove sparse flying samples.
It cannot remove a dense incorrectly selected background patch.

**Aim and box center are different:**
- The default aim is an actual retained surface sample near the cloud's median.
- A manual aim click uses that pixel's depth and camera calibration.
- The automatic box encloses ALL retained sampled points. It describes partial
  visible surfaces, not the complete hidden object volume.

Open **Adjust the 3D box** to edit center XYZ, size XYZ and upright yaw around map Z.
Click Apply 3D box. Manual edits are marked as human estimates. This can pad an
almost planar observation or express your intended full-object boundary, but it does
NOT reconstruct hidden points, crop the source cloud, or establish ground truth.

The UI reports how many observed samples fall inside a manually adjusted box.
A box can be drawn through a wall in projection; projected guides are not occlusion
aware. Use another view for a more reliable spatial annotation.

## 5. Label and save

Choose **New object**, enter a label such as `red vessel` or `gray cabinet`, and
optionally a category/note. Click **Save labelled object**.

You receive a stable ID, e.g. `manual_0001`. There is no inferred manufacturer,
maintenance identity or workbook asset association. Overlapping object annotations
are allowed; no pixel or point is forced to belong to exactly one label.

### Add a second view to the same object (recommended)

1. Resume live, move to a second useful viewing position, then Freeze.
2. Outline the SAME physical object and build a new selection.
3. Under Destination choose the existing ID, e.g. `manual_0001`.
4. Inspect the label and click Save.

The new samples are combined in the common map frame (1 cm voxel deduplication).
Every observation and prior revision is retained independently. Automatic bounds
are recalculated from the accumulated visible samples. An existing human-adjusted
box is preserved unless you deliberately apply another manual box. The newest view's
aim point becomes the object's latest aim point.

This operation trusts your manual identity choice; it does NOT run learned association
or correct inconsistent Gaussian depth across views. Check the combined box and evidence.

Select a saved item to reuse its ID/category/note. To rename without adding a view,
use **Update selected label / note**. Source geometry is not changed by renaming.
To create the next object, explicitly choose **New object** under Destination.

## Outputs (all under the new --out folder)

```text
outputs/manual_labels/scene_v1/
  source.json                         # Source scene/map hashes and original transform
  annotations.json                    # Main consolidated object-label manifest
  annotations.csv                     # Compact spreadsheet-friendly summary
  objects/manual_0001/
    object.json                       # Latest object label, target, box and provenance
    observations/view_<timestamp>_<id>/
      rgb.png                         # Raw exact frozen image, unchanged
      depth_scene.npy                 # Raw expected Z in original scene units
      depth_m.npy                     # Same depth in provisional metres
      alpha.npy
      camera.json                     # K, c2w/w2c, c2map/map2c, frame identity/pose
      mask_drawn.png                  # Original manual rectangle/polygon
      mask_depth_supported.png         # After alpha/depth/connected checks
      overlay.png                     # Display-only tint, projected box and aim
      object_crop.png                 # Tight RGBA crop; alpha is supported mask
      samples_map.npz                 # Points/colors/UV before & after SOR
      surface_map.ply                 # Ordinary XYZRGB points, not original Gaussians
      observation.json
      complete.json                   # Checksums for observation files
    revisions/rev0001/
      cloud_map.ply                   # Combined ordinary XYZRGB cloud in map axes
      cloud_map.npz
      cloud_scene.ply                 # Same surface samples in original GS axes
      object.json
    revisions/rev0002/                # Created by adding views; previous revision stays
      ...
```

An object record includes label/category/note, target coordinates in BOTH frames,
3D box center/extents/yaw, box corners in BOTH frames, observed-surface bounds,
reference camera pose, observation list, and current cloud paths.

`verified_navigation_goal` stays null. The object aim is not an approach/stopping
position; a future navigation application must find a reachable position nearby.
A reference camera position also is not automatically a verified walking position.

The original frame conventions are retained:

```
x_map = s * x_GS
y_map = s * z_GS
z_map = s * (floor_y - y_GS)
```

With the current bundle, s=1.43 and floor_y=1.1. Backprojection uses the renderer's
pixel centers `(column + 0.5, row + 0.5)`. Scale applies to positions and depth, not
rotation matrix columns. Exported `cloud_scene.ply` is NOT an original Gaussian subset.

## What to share for the final demonstration

For a lightweight initial check, share `annotations.json`, `annotations.csv`, and
one object folder (especially its overlay/crop and object.json).

To package all labels and evidence, first stop the server, then:

```bash
bash manual_labeler/run.sh export \
  --out outputs/manual_labels/scene_v1 \
  --zip outputs/manual_labels/scene_v1_export.zip
```

This ZIP includes generated annotation work only, not the original full Gaussian
PLY, model checkpoints, or other demo outputs. It may still contain Valmet scene
images; keep the package within the permitted team/company sharing scope.
A pre-existing ZIP is not overwritten; choose a new name for a later export.

The header's **Export labels JSON** button downloads only the consolidated manifest.
The final web demo is not changed or automatically updated in this stage.

## Fallbacks

Lower rendering load (new launch, same labeling dataset is permitted):

```bash
bash manual_labeler/run.sh serve \
  --bundle outputs/service_demo/main_v1 \
  --out outputs/manual_labels/scene_v1 \
  --width 480 --height 360 --render-hz 1 --open-browser
```

Map/controls only, without CUDA:

```bash
bash manual_labeler/run.sh serve --out outputs/manual_labels/scene_v1 --no-render --open-browser
```

Label an already-saved RGB/depth/camera set without CUDA:

```bash
bash manual_labeler/run.sh serve \
  --bundle outputs/service_demo/main_v1 \
  --out outputs/manual_labels/scene_v1 \
  --snapshot-prefix outputs/renders/view_A \
  --open-browser
```

The prefix requires `_rgb.png`, `_depth.npy` in scene units, `_alpha.npy`, and
`_camera.json`. A depth-visualization PNG alone is insufficient. Snapshot mode is
static: joystick movement is disabled and the actual saved camera pose is used.

If CUDA fails, read the terminal and renderer status. Do not reinstall/update the
working environment automatically. Stop other GPU-heavy processes; this module
loads the original scene once and uses `spawn`, not `fork`, for its CUDA worker.

Port conflict: close the old labeler or pass `--port 8773`. Only one writer may use
an output directory. The local server binds to 127.0.0.1 and validates local Host,
Origin and a POST token. It is a single-user demo tool, not a production/public API.

## Focused validation

- 8 CPU tests: scene/map roundtrip, rigid camera transforms, image reprojection,
  seed depth isolation, invalid seeds, polygon/aim selection, upright box corners,
  immutable observation exports, multiple-view append, and source-identity refusal.
- Python compilation and JavaScript syntax checks.
- Browser smoke at 1440x900 plus viewport check at 1366x768: freeze displayed frame,
  mouse-drawn rectangle, seed selection, 3D preview, save, resume, and labels export.
  A synthetic scene and actual local HTTP server were used. Container browser network
  policy required a test-only request bridge; delivered server checks are unchanged.
- Actual live CUDA rendering/joystick response on the Valmet scene remains to be
  checked on the user's laptop. Renderer calls reuse the earlier working interface.

## Depth limitation / primary references

Expected Gaussian Z depth is an alpha-weighted estimate, not a physical range-sensor
measurement. Single-view boxes enclose observed samples. Additional views and manual
review help, but do not guarantee accurate complete object geometry.

- gsplat 1.5.3 rasterization and expected-depth definition:
  https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- Python 3.10 spawned multiprocessing:
  https://docs.python.org/3.10/library/multiprocessing.html
