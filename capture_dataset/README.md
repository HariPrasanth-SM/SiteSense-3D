# Valmet — Stage 1: calibrated multi-view capture

This adds a separate `capture_dataset/` folder beside your existing `avatar_demo/`.
It prepares camera positions and renders RGB, expected depth, alpha, validity masks,
and exact intrinsics/poses from the original Gaussian PLY. It does **not** change
navigation, edit the occupancy map, install models, or run segmentation/association.

## Scope and important limits

- Intended for the existing Ubuntu 22.04, Python 3.10, gsplat 1.5.3 environment.
- Reuses `avatar_demo.core.FloorMap`, `camera_matrices`, and the working
  `avatar_demo.render_worker._load_scene` Gaussian loader. Keep that folder present.
- Original map and Gaussian PLY are read-only. No changes to ROS, CUDA, `.bashrc`,
  `run.sh` in the avatar package, or renderer source are made.
- The new launcher uses your `.venv`, preserves its existing library paths, and sets
  `CUDA_HOME=/usr/local/cuda-12.6` only when no value exists. It does not install dependencies.
- A selected gray camera location grants **capture permission only**. It is not a
  free-space/navigation annotation, and not a verified 3-D collision-free location.
- These are independent virtual viewpoints, not a video trajectory or traversed route.
- Sampling positions throughout the floor map does not guarantee all objects are visible.
- Floor remains a flat plane at Gaussian Y=1.1; scale remains an assumed 1.43 m/scene unit.

## Install

Save `valmet_capture_stage1.zip` in Downloads, then:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_capture_stage1.zip -d .
```

This is a new module, so it leaves the existing avatar application unchanged. It
uses NumPy, SciPy, Pillow, PyYAML, Tkinter, plyfile, torch, and gsplat already used
by the project. Do **not** reinstall/upgrade working torch/gsplat for this step.

## 1. Select positions and prepare the dataset plan

```bash
bash capture_dataset/run.sh select \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --out outputs/datasets/valmet_views_v1 \
  --positions 40
```

The map comes from your YAML/PGM, not from a screenshot. Known-free cells are pale
green, occupied cells dark, and unknown cells gray. The zero-extra-clearance avatar
setting did not modify the map file, so all input-free cells are available here.

Controls:

- **Left click:** pin a stationary camera. It snaps to the selected map cell center.
- **Allow clicked GRAY cells for capture ONLY:** enable before pinning gray corridor
  positions. Gray cells are NEVER added automatically.
- **Right click near a pin:** remove it.
- **Shift+left click:** exclude a 0.4 m disk from automatic camera sampling. Remove
  pins there first. This affects capture planning only, not occupancy.
- **Undo exclusion:** remove the latest exclusion.
- **Recompute spread:** regenerate automatic positions.
- **SAVE PLAN + CLOSE:** commit the plan and close the picker.

Suggested first selection: pin 5–8 positions along the gray corridor you inspected,
spaced along its length, plus any particularly useful known viewpoints. The tool
fills the **remaining** budget with spatially spread known-free positions. Forty
positions means pins plus automatic positions, not forty plus the pins.

Auto positions use a small 0.10 m map-center clearance, NOT the avatar's 0.25 m radius,
and a 0.50 m spacing target. These are provisional-metre sampling heuristics. Pins
are exempt from the auto-clearance requirement but cannot be in occupied cells.
If fewer than 40 positions satisfy the settings, the tool does not invent more:
it reports the actual count. A new plan can use `--clearance 0` or a smaller
`--spacing`, but cameras near surfaces need visual review.

### Default image plan

| Parameter | Value |
|---|---|
| Stations | Up to 40 total |
| Yaw | 0, 60, 120, 180, 240, 300 degrees |
| Pitch | 0 and -20 degrees; negative looks down |
| Camera height | 1.7 provisional m |
| Resolution | 640 x 480 |
| Vertical FOV | 70 degrees |
| Appearance | SH degree 3, same convention as the avatar renderer |
| Total | 40 x 6 x 2 = 480 images when 40 stations are available |

Positions and rotations are known because these cameras are generated from the
existing scene. There is no COLMAP/reconstruction step.

View the planned camera distribution:

```bash
xdg-open outputs/datasets/valmet_views_v1/camera_positions.png
```

Purple stations are manually pinned unknown cells; gold are pinned free cells;
blue are automatic free-cell locations. `P` marks pilot stations. Station numbers
are identities, not navigation order.

## 2. Render only the pilot first

Close the avatar application, RViz, and other GPU-heavy jobs before capture. The
same 1.68M-Gaussian scene would otherwise be duplicated in GPU memory.

```bash
bash capture_dataset/run.sh render \
  --out outputs/datasets/valmet_views_v1 \
  --pilot
```

With the defaults, the pilot selects up to six stations (including up to three
manual pins), three yaw directions per station, and both pitches: normally **36
images**. It does not simply render the first 36 frames from one part of the room.

For a single-frame smoke test, add `--limit 1`. Repeating the command resumes
remaining pilot frames. A full render later reuses the pilot frames.

The original Gaussian PLY is loaded once, all Gaussians retained except invalid
parameters as in your existing loader, and one camera rendered at a time using
`torch.inference_mode()`, `RGB+ED`, and `packed=True`. No model training is involved.
No FPS/time promise: actual throughput and per-frame write time are printed.

## 3. Inspect the pilot and STOP at this checkpoint

```bash
xdg-open outputs/datasets/valmet_views_v1/qa/index.html
```

This is a local static contact-sheet page, not a web server. It opens RGB and depth
contact sheets without uploading the industrial data. Review:

- Are views upright and reasonably centered on equipment?
- Are gray-corridor stations inside the corridor rather than inside a pipe or tank?
- Are machines visible at useful sizes from multiple DIFFERENT stations?
- Do near and far surfaces look plausible in depth?
- Are there wholly obstructed or fragmentary/unreconstructed views?

`quality.csv` has heuristic warnings (low rendering coverage, too much very-near
geometry, or little RGB variation). No frames are automatically removed. Alpha is
composited coverage, NOT geometric confidence. A score cannot prove a view is valid.

Please share `camera_positions.png`, `qa/contact_rgb_001.jpg`, and
`qa/contact_depth_001.jpg` at the first checkpoint. Add `quality.csv` and an original
full-resolution example frame if useful. The original 370MB PLY need not be uploaded.

## 4. After the pilot is accepted: finish all frames

This is the later command; not required before reviewing the pilot.

```bash
bash capture_dataset/run.sh render \
  --out outputs/datasets/valmet_views_v1
```

Each completed frame has a completion marker with checksums. Re-running verifies
and skips complete matching frames. Partial/corrupted frames are rerendered. Ctrl+C
leaves completed frames intact. Only one render process may write a dataset.

The scene hash, dependency-source hashes, torch/gsplat versions, and plan hash are
recorded. Resume refuses a changed scene, rendering configuration, or key software
version rather than silently mixing incompatible records.

Raw float arrays alone for 480 x 640 x 480 frames (depth_scene, depth_m, alpha) occupy
about 1.65 GiB, plus RGB/validity PNGs, previews and metadata. Keep at least **4 GiB**
free for this first dataset, and considerably more for subsequent mask datasets.
Disk availability is checked conservatively before rendering.

## Revise a bad plan without overwriting it

Once saved, the plan is immutable. Create `valmet_views_v2` to change cameras,
resolution, FOV, scale or SH degree. Editing `plan.json` manually invalidates its
hash. Keeping old datasets also preserves downstream references.

To reuse existing pins and exclusions without opening the picker:

```bash
bash capture_dataset/run.sh plan \
  --selection outputs/datasets/valmet_views_v1/selection.json \
  --out outputs/datasets/valmet_views_v2 \
  --positions 40
```

This requires the same map identity. For a new visual selection, run `select` with
a new output directory and click the desired locations again. Source map and
previous captures remain intact.

If GPU memory is insufficient, create a NEW plan with lower resolution, for example
`--width 480 --height 360`. Do not change the rendering recipe halfway through one
dataset. SH degree 0 is an optional new-plan fallback, but reduces directional
appearance quality. Do not install another renderer just for this batch.

## Files

```text
outputs/datasets/valmet_views_v1/
  plan.json                  # Immutable full camera/configuration plan + hash
  selection.json             # Explicit capture-only pins/exclusions from picker
  camera_positions.png       # Map + station numbers and pilot markers
  camera_positions.csv
  frames.jsonl               # All planned view records
  render_binding.json        # Source scene/code hashes
  render_environment.json    # torch, gsplat, GPU, configuration identifiers
  index.jsonl                # Joinable frame IDs and paths, completion status
  quality.csv
  summary.json
  frames/
    s000_y00_p00/
      rgb.png                # Original-size RGB uint8, lossless PNG
      depth_scene.npy        # RAW float32 expected camera-Z depth, scene units
      depth_m.npy            # Derived float32 = depth_scene * 1.43
      alpha.npy              # RAW float32 composited coverage
      valid.png              # Separate adjustable-threshold validity mask
      camera.json            # K, c2w/w2c, c2map/map2c, frame conventions, pose
      qa.json                # Warnings and measured render time
      complete.json          # Checksum completion marker, written LAST
  qa/
    index.html
    contact_rgb_001.jpg
    contact_depth_001.jpg
    ...
```

`rgb.png`, raw depth, and `alpha.npy` are not modified based on quality scores.
`valid.png` is merely a convenience and can be regenerated from the raw data later.
The saved depth is NOT the normalized grayscale thumbnail.

The camera matrices have two separate metric conventions:

- `camera_to_world` / `world_to_camera`: original Gaussian scene coordinates,
  used with **scene-unit** depth.
- `camera_to_map` / `map_to_camera`: rigid rotation and translation in provisional
  metres, used with **metre** depth. Scale is NOT embedded in their rotation columns.
- `scene_to_map` stores the similarity transform between the scene and map.
- Camera axes: OpenCV +X right, +Y down, +Z forward.
- `pixel_sampling` describes the rasterizer pixel centers (column+0.5, row+0.5).
- Depth is expected camera-axis Z, not ray length, and is not physical sensor data.

A frame ID such as `s012_y03_p01` means station 12, yaw index 3 (180 degrees under
the defaults), pitch index 1 (-20 degrees). Pair it with the dataset plan hash to
obtain a globally unambiguous key.

## Rebuild reports without CUDA

```bash
bash capture_dataset/run.sh report --out outputs/datasets/valmet_views_v1
```

## CPU tests

```bash
source .venv/bin/activate
python -m unittest discover -s capture_dataset/tests -v
```

Tk UI tests require a display and are skipped if no DISPLAY is present. No CUDA
or scene file is required for these synthetic tests. See `VALIDATION.md`.

## Sources for the API conventions

- gsplat 1.5.3 rasterization: https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- DINOv3 official implementation, for later features/text-alignment distinction:
  https://github.com/facebookresearch/dinov3
- FastSAM official integration, for the next stage:
  https://docs.ultralytics.com/models/fast-sam/

## Not implemented in this stage

FastSAM mask extraction, DINOv3 features, text search, cross-view association,
instance point clouds and interactive semantic scene overlays are deliberately
not run yet. See `DOWNSTREAM_CONTRACT.md` for the intended identifiers and data
relationships. Review the pilot before investing in those stages.
