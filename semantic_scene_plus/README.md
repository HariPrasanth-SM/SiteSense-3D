# Valmet Stage 4C — clean object clouds, original Gaussian subsets, and evidence UI

Additive module: `semantic_scene_plus/`. Keep `semantic_scene/`, `avatar_demo/`, all upstream modules,
and all existing data. Nothing here reruns segmentation, DINOv3, association, training, or reconstruction.
No changes to CUDA, PyTorch, gsplat, ROS, the launcher of the avatar, or shell configuration.

## What is different

1. Statistical outlier removal (SOR-style) on EACH region's fused `surface.npz`.
2. Raw and cleaned clouds both remain available; removed points are exported separately.
3. Displayed bounding frames change with the selected representation. Cleaned boxes enclose ALL retained
   points, not an unannounced percentile crop. They are still partial-observation boxes, not full assets.
4. Optional extraction of original Gaussian records: XYZ, log scales, quaternions, opacity logits, all
   SH coefficients and other vertex properties are copied unchanged from the source PLY.
5. Original Gaussian row IDs and supporting-view votes are saved. Regions can overlap.
6. Viewer cards now show frame ID, mask index, station ID, crop, and a translucent mask on the full image.
7. An optional GPU button renders the selected Gaussian subset from the current browser 3D viewpoint.
   This is actual gsplat rasterization, not just plotting Gaussian centers.

## Installation

Save `valmet_semantic_scene_plus.zip` to Downloads; close an older viewer using port 8765 first.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_semantic_scene_plus.zip -d .
source .venv/bin/activate
```

Uses existing NumPy, SciPy, Pillow and plyfile. GPU rendering additionally uses your EXISTING torch,
gsplat and `avatar_demo.render_worker._load_scene`. No pip upgrades or new model downloads required.
The new launcher explicitly uses `.venv/bin/python` when present, without changing CUDA/library paths.

## 1. Prepare cleaned clouds and Gaussian subsets

Use the actual scene path containing `scene.json` and `regions/`. The earlier 150-seed command used:

```bash
SCENE="outputs/semantic_scene/valmet_regions150_v1"
ENH="outputs/semantic_enhance/valmet_regions150_v1"

bash semantic_scene_plus/run.sh prepare \
  --scene "$SCENE" \
  --out "$ENH" \
  --gaussian-ply "data/Industrial Boiler Room.ply" \
  --neighbors 24 \
  --std-ratio 2.0 \
  --splat-radius 0.08 \
  --min-splat-views 2
```

Optional smoke run: append `--limit 3` to prepare three NEW regions. Run the same command without
`--limit` afterward to finish; completed matching regions are verified and reused. `--region ID`
processes a single exact region ID. Do not modify the source masks/capture while preparing.

The original PLY is loaded on CPU, once per prepare command. No neural model or CUDA computation
is used. Subsequent settings changes need a NEW `--out` folder; no raw cloud is overwritten.

### SOR settings

For each point, compute the mean distance to its 24 nearest OTHER neighbors. Remove points above
`global mean of those distances + std_ratio * global std`. Each region is handled independently.
This is the usual SOR-style criterion but is not claimed numerically identical to Open3D's implementation.

- `--std-ratio 2.0`: starting setting.
- `--std-ratio 1.5`: stronger rejection, more risk to thin/sparsely sampled features.
- `--std-ratio 3.0`: gentler rejection; retains more points.
- No "largest component only" rule: separated genuine components are not removed merely for being separate.
- Nonfinite points are removed. Too-small resulting clouds cause a clear error rather than a fake box.
- Removing >20% emits a warning in the viewer and per-region report.

A compact but incorrect attached surface may survive SOR. SOR is NOT an object-identity correction,
semantic merge correction, or a guarantee of geometric accuracy.

### Gaussian subset selection

1. Convert original Gaussian CENTERS to the existing map frame for comparisons only.
2. Propose centers within 0.08 provisional metres of the cleaned surface samples, using a bounding gate
   and nearest-neighbor search. This is not nearest-object ownership or a full-scene exclusive segmentation.
3. Reproject candidates into the region's original member views.
4. Require the projected center inside the corresponding mask (dilated by one pixel), alpha >=0.5,
   and camera-Z agreement within `0.12 + 0.03 * depth` provisional metres.
5. Require support from at least two distinct station IDs (the selected member views in Stage 4B).
   Outside/occluded candidates simply do not get a vote. There is no automatic single-view fallback.
6. Copy the accepted records from the original PLY and retain their original row indices.

This is APPROXIMATE original-splat extraction. A Gaussian center is not necessarily on the physical
surface, expected depth can mix surfaces, and center projection is not exact per-pixel contribution tracing.
Thin parts/back surfaces can be incomplete; unrelated nearby geometry can leak in. Original SH appearance
is preserved but removing neighboring Gaussians can change compositing, shadows, transparency or halos.

Do not delete low-opacity Gaussians simply because they are faint: default export does NOT opacity-prune.
Only nonfinite parameters and invalid quaternions are automatically rejected. An optional
`--max-gaussian-sigma 0.25` rejects Gaussians with any activated axis sigma >0.25 map metres;
this is OFF by default, must use a new output folder, and may remove genuine appearance contributions.

### Tuning when splats are sparse or leaky

Try the defaults first. In a NEW output folder:

- Sparse/incomplete subset: `--splat-radius 0.12`; optionally `--min-splat-views 1` for a visibly flagged,
  less reliable demo subset. One-view support increases leakage risk.
- Leakage onto nearby equipment/wall: `--splat-radius 0.05`, keep at least two supporting stations.
- Ghostly large splats: consider the optional sigma cap above; do not assume every large splat is wrong.

These are starting heuristics under scale 1.43, not calibrated physical thresholds. An empty subset is
reported as zero selected Gaussians; its cleaned surface remains usable. No made-up splats are produced.

## 2. Launch the enhanced browser

```bash
bash semantic_scene_plus/run.sh view \
  --enhancement "$ENH" \
  --environment outputs/ros/valmet_colored_cloud.npz \
  --live-splats \
  --port 8765 \
  --open-browser
```

Keep this terminal running. Browser address: `http://127.0.0.1:8765`.
No ROS/Gazebo/RViz process is required. Stop other large GPU jobs for the first splat-render test.
Use your full environment NPZ with `--max-env-points 350000` if needed; the source file stays unchanged.

### Geometry controls

**Surface display**:
- Cleaned surface points + compact box (default).
- Raw surface points + original box.
- Raw voxels with >=2 station support (as before).
- Original Gaussian centers (point display, NOT splat rendering).
- SOR-removed points (diagnostic; may include valid sparse parts).

Select a region; use **Focus selected**, **Isolate selected region**, and hide **Environment context**
for a clearer comparison. Point size/colors remain adjustable. Boxes reflect the selected geometry.
No second scaling is applied to the environment NPZ, which must already be in map coordinates.
Unprocessed regions from a `--limit` run remain available as raw clouds in the cleaned mode.

### Actual Gaussian rendering

1. Select a region with a nonzero splat count.
2. Press **Focus selected** so the region occupies a useful part of the camera view.
3. Orbit/zoom using the 3D panel on the left.
4. Press **Render selected splats from current 3D view**.
5. A real alpha-composited gsplat rendering appears in the right panel; use **Save this splat image**.

Move the left camera and press the button again to request another view. This is ON-DEMAND GPU rendering,
not continuous in-browser Gaussian rendering. Only the selected subset is rendered; there is no full-room
occlusion in that image. The GPU keeps the most recently selected subset, rather than the whole room
and all objects simultaneously. Browser changes during a request do not change the recorded rendering pose.

First request can initialize or JIT-build the same installed gsplat CUDA extension. No FPS guarantee.
If a render fails, the point-cloud viewer stays available and shows the error. The terminal logs the cause.
Do not reinstall PyTorch/CUDA blindly. Omit `--live-splats` for a CPU-only cloud/evidence viewer.

### Evidence panel

Every card shows a visible frame ID, mask ID and camera-station ID, plus crop AND full-image overlay.
Click the card. The larger evidence image can show:
- Full image + translucent mask.
- Original full image.
- Context crop.
- Masked crop.
- Gaussian subset rendered from this source camera (after optional pre-rendering below).

Click the larger image to open full resolution. Camera metadata/observation IDs remain associated;
the cyan frustum represents the saved source camera. Existing manual review labels and suggestions load
as before. Saving a manual review writes only the ORIGINAL scene's `review.json`/history, using the existing
review API; it never rewrites original surface geometry or association records.

## 3. Optional: pre-render source-camera splats for reliable presentation

Useful when you do not want live GPU requests during the demo:

```bash
bash semantic_scene_plus/run.sh render-previews \
  --enhancement "$ENH" \
  --views 3 \
  --width 640
```

`--limit 3` restricts the pass to three region folders; `--region region_...` selects one region.
The renderer loads one subset at a time, and skips existing matching previews. No whole-room model required.
Refresh the browser and select **Splat at this source camera**. First three views per region have previews;
other views state that no preview exists. Increase `--views` as needed (up to 32).

This is optional: live rendering works without generating these previews.

## Outputs and coordinate conventions

```text
outputs/semantic_enhance/valmet_regions150_v1/
  config.json
  summary.json
  regions/region_.../
    enhancement.json
    complete.json
    surface_clean.ply          # XYZRGB, map frame
    surface_clean.npz          # plus original surface row IDs and observation membership
    surface_removed.ply        # diagnostic only, map frame
    surface_removed.npz
    gaussian_indices.npy       # original vertex row indices, int64
    gaussian_centers_map.npz   # centers transformed for map display + RGB, support votes
    gaussians_original.ply     # original Gaussian frame, full original attributes
    views/v00_overlay.jpg      # original full RGB + mask tint/outline
    renders/v00.png            # optional real Gaussian source-camera rendering
    renders/v00.json           # exact source and camera matrices
```

IMPORTANT: `surface_clean.ply` is an ordinary XYZRGB point cloud; `gaussians_original.ply` is a real
Gaussian PLY. Do not load the former as a splat model. Do not transform only the latter's XYZ into the map
frame: orientations/covariances/directional SH would then be inconsistent. The viewer converts cameras
into the original Gaussian frame instead. The transformation is recorded in config:

```
x_map = 1.43 * x_GS
y_map = 1.43 * z_GS
z_map = 1.43 * (1.1 - y_GS)
```

Gaussian center bounds are NOT full ellipsoid/splat extents. Cleaned boxes enclose retained observed
surface samples, not the full hidden object, navigation approach point, or collision-safe equipment volume.
Exports can overlap across hypotheses; regions are not forced to own points exclusively.

## Preservation, resume, rollback

- Original scene/geometry/labels/captures are never rewritten by `prepare`.
- Stamped completed regions are verified and reused when settings match.
- Settings/scene changes require a new output folder.
- Incomplete region files without complete.json are recreated on retry.
- Labels edited in the viewer use the original scene's review layer, so old and new viewers agree.
- To roll back, simply start the original `semantic_scene/run.sh view` against the original scene.
- All raw point data remain in the original scene even after removing individual enhanced outputs.

## Validation performed for this delivery

Focused checks only, per the requested fast turnaround: Python compilation, JavaScript syntax,
synthetic SOR behavior, scene/map camera transforms, projection depth gates, viewer manifests and
clean/raw/removed/Gaussian-center HTTP buffers, source-image and overlay endpoints.
No actual Valmet PLY, CUDA rendering, or learned models were executed here. Your laptop is the first
full original-Ply extraction and GPU rendering test. These checks do not establish segmentation accuracy.

## References

- SOR principle (Open3D): https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
- gsplat 1.5.3 rasterization: https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- PLY structured records: https://github.com/dranjan/python-plyfile
