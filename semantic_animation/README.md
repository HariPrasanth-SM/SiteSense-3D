# Valmet — Gaussian scene → points → separated regions → object splats

An additive **offline animation** tool. It uses the existing Stage 4C enhanced
regions, their real original Gaussian subsets, the original room Gaussian PLY,
and an optional map-frame XYZRGB environment NPZ. No training, segmentation,
association, ROS, Gazebo, browser automation, or CUDA installation is performed.

## What it generates

Default: **18 seconds, 24 fps, 1280×720 = 432 frames**.

1. Hold a real Gaussian rendering of the original industrial environment.
2. Crossfade to the room's point-cloud representation in the same camera frame.
3. Highlight a selected set of colored region point clouds and fade room context.
4. Translate those region clouds into separate gallery positions while pulling back the camera.
5. Stagger each region's point-cloud fade into its **original Gaussian subset**.
6. Hold the labelled gallery.

This is an **illustrative exploded presentation**, not physical motion, a
collision-free path, new reconstruction, or proof of full-scene segmentation.
The original overlapping hypotheses remain independent.

The reference MP4 was inspected: it shows an industrial scene, a scanning-style
reveal and colored labelled objects. The exploded gallery is the requested
extension, not a claim that the reference contains this exact motion.

## Important representation distinction

- `surface_clean.ply`: XYZRGB visible-surface samples in the project's map frame.
- `gaussians_original.ply`: real original Gaussian records in the original Y-down
  Gaussian frame, including scale, quaternion, opacity and spherical harmonics.
- The point-cloud shots display actual point-cloud samples as small isotropic
  screen-sized dot sprites using the same renderer. Those artificial display
  sprites are NOT saved as original reconstructed Gaussians.
- The final object shots use the original anisotropic Gaussian scales,
  orientations, opacity and SH appearance. No points-only fallback is silently
  substituted when a region has no Gaussian subset.
- During the reveal, opacities are animated for presentation. At the end, original
  object Gaussian opacity is fully restored and point sprites are fully hidden.

## Installation — preserve the existing project

Close the avatar, old semantic viewer's live splat renderer, and other GPU-heavy
jobs for the first test. Save the ZIP in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_semantic_animation.zip -d .
source .venv/bin/activate
```

Keep `avatar_demo/` (the working Gaussian loader) and your enhanced region files.
The new code lives only in `semantic_animation/`. It does not change prior modules,
source geometry, maps, captions/review files, CUDA variables, or package versions.
Use the existing NumPy, SciPy, Pillow, plyfile, torch and gsplat installation.

FFmpeg is needed only for MP4 encoding:

```bash
ffmpeg -version
```

If absent, install **only** that package:

```bash
sudo apt install ffmpeg
```

No full system upgrade or driver installation is needed.

## 1. Prepare an animation with a small set of good objects

```bash
ENH="outputs/semantic_enhance/valmet_regions150_v1"
ANIM="outputs/animations/valmet_reveal_v1"

bash semantic_animation/run.sh prepare \
  --enhancement "$ENH" \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --out "$ANIM" \
  --objects 12 \
  --duration 18 \
  --fps 24 \
  --width 1280 --height 720 \
  --portable
```

Use the actual enhancement output directory from Stage 4C if its name differs.
A smaller environment file `outputs/ros/valmet_colored_cloud.npz` also works.
Omit `--environment` to derive a visualization point cloud from the original room
Gaussian centers and DC appearance coefficients. No change to the source PLY.

The room PLY is read from the enhanced scene's `config.json`. Supply `--scene-ply`
only to give the same original source file at a different path; its hash is checked.

`--portable` also copies the whole-room PLY into the animation assets. With that
flag, the animation's input geometry and camera records are self-contained. Without
it, the approximately 370 MB room file is referenced in its original location.
Do not redistribute Valmet data without permission.

### Why 12 regions rather than every region?

A 10–20 second animation is easier to inspect with 6–16 coherent regions. The
150-seed run may include walls, partial assemblies and duplicates, not 150 distinct
assets. Automatic selection is only a presentation heuristic:

- Excludes manually rejected regions and regions without at least 20 original splats.
- Prefers human-accepted equipment/parts and multi-view-supported regions.
- Omits labelled structural regions and default max extent >5 map metres.
- Avoids selecting pairs sharing more than 65% of the smaller Gaussian-ID set.
- Does not delete or merge any region in the original data.

Unlabelled items are `R01`, `R02`, etc. Suggested labels retain `?`; reviewed labels
are copied as a presentation snapshot, never promoted silently.

### Review the selection

```bash
xdg-open "$ANIM/selection.html"
xdg-open "$ANIM/selected_assets.jpg"
```

The HTML includes all available region IDs. Use explicit IDs in a **NEW** output
folder to choose your strongest examples:

```bash
bash semantic_animation/run.sh prepare \
  --enhancement "$ENH" \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --out outputs/animations/valmet_reveal_selected_v2 \
  --regions region_ID_A,region_ID_B,region_ID_C \
  --duration 18 --fps 24 --width 1280 --height 720 \
  --portable
```

Replace those placeholder IDs with actual IDs shown in `selection.html`. Explicit
selection bypasses automatic structural/size/overlap heuristics. It still requires
valid prepared Gaussian subsets and excludes manually rejected regions.

## 2. Render a ten-frame storyboard FIRST

```bash
bash semantic_animation/run.sh render --out "$ANIM" --preview
xdg-open "$ANIM/storyboard.jpg"
```

Or open the individual full-resolution frames through:

```bash
xdg-open "$ANIM/preview.html"
```

These are actual timeline frames and are reused by the full render. Check:

- Is the opening view sharp and recognizable?
- Do the objects really look like useful regions, not wall fragments?
- Are the gallery objects large enough and not clipped?
- Are the original Gaussian subsets coherent when rendered alone?

### Choose a known good opening camera

The default starting camera is selected from saved region-supporting views using
approximate selected-center coverage. That is not a visibility guarantee. Use a
known good saved camera in a new plan if needed:

```bash
bash semantic_animation/run.sh prepare \
  --enhancement "$ENH" \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --camera outputs/renders/view_A_camera.json \
  --out outputs/animations/valmet_reveal_camera_v2 \
  --objects 9 --duration 18 --fps 24 --width 1280 --height 720 \
  --portable
```

The loader accepts either `camera_to_map`, or `camera_to_world` in the original
Gaussian scene axes. The animation uses its own saved 55° vertical FOV; it does
not pretend to retain the original camera intrinsics. `--fov` changes that value
when preparing a new plan.

The whole original room is loaded, but a single initial perspective cannot show
all surfaces simultaneously. The moving cinematic camera can cross walls while
room context is fading; it is not an avatar trajectory or robot route.

## 3. Render the full movie

```bash
set -o pipefail
bash semantic_animation/run.sh render --out "$ANIM" \
  2>&1 | tee -a "$ANIM/render_console.log"
```

The renderer loads source assets persistently, processes one camera at a time,
and writes each frame immediately. It keeps no full video tensor in GPU memory.
The original room parameters are released after the opening transition.

Completed frames have SHA256-backed metadata. Repeating the same command reuses
complete matching frames; interrupted/changed frames are recreated. A file lock
prevents two writers. Resume refuses edited plans or incompatible code/runtime
versions. Use a new output directory for a changed camera/layout/duration/size.

For a bounded pass of at most 30 **new** frames:

```bash
bash semantic_animation/run.sh render --out "$ANIM" --limit 30
```

Leave approximately **6 GiB free** for copied inputs, two PNG sequences, and MP4s.
Actual compressed sizes depend on the scene. This is offline rendering: 24 fps is
the VIDEO playback rate, not a required live rendering rate. No laptop throughput
or GPU memory guarantee is made. This requires no model training.

## 4. Encode the videos

```bash
bash semantic_animation/run.sh encode --out "$ANIM" --both
xdg-open "$ANIM/semantic_reveal.mp4"
```

Outputs:

- `semantic_reveal.mp4`: captions plus region labels.
- `semantic_reveal_clean.mp4`: same render without text for editing.

Both are silent H.264 MP4, yuv420p, CRF 18, with fast-start metadata. The encoder
checks every expected frame; it will not silently produce a truncated video.
Audio/music is not copied from the reference. Add only appropriately licensed
sound in your editor if needed.

Equivalent manual FFmpeg input convention:

```bash
ffmpeg -framerate 24 -i "$ANIM/frames/%06d.png" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart video.mp4
```

The supplied `encode` command uses the planned FPS automatically and also checks
frame identity/integrity; prefer it over a hardcoded manual command.

## Saved work products

```text
outputs/animations/valmet_reveal_v1/
  plan.json                   # Source identity, exact selections, intrinsics, layout, settings
  timeline.json               # Stage timing
  motion.npz                  # Every camera pose, per-object translation, reveal/fade weights
  camera_path.jsonl           # Same poses + map and original-Gaussian object offsets per frame
  selection_report.json       # Included/omitted region IDs with reasons
  selection.html
  selected_assets.jpg
  assets/
    environment_original.ply   # Original full scene when --portable is set
    environment_points_map.npz
    environment_points_map.ply
    region_.../
      asset.json
      region.json
      enhancement.json
      surface_clean.ply       # Full cleaned original surface, MAP frame
      surface_clean.npz       # Original provenance retained
      display_points_map.npz  # Deterministic display-only subsample
      gaussians_original.ply  # Original extracted records, original GS axes
      gaussian_indices.npy    # Original scene row IDs, if available
      gallery_surface_map.ply # Final translated surface in MAP coordinates
      gallery_gaussians_original_axes.ply # Final translated real splats in GS axes
      evidence.json
      evidence/               # Up to three crops/context/overlays + pose records
  frames_clean/000000.png ...
  frames/000000.png ...        # Captioned counterpart
  frame_metadata/000000.json ...
  render_environment.json
  render_summary.json
  storyboard.jpg
  preview.html
  semantic_reveal.mp4
  semantic_reveal.json        # Encoder settings, hashes, duration
  semantic_reveal_clean.mp4
  semantic_reveal_clean.json
```

Use these point clouds, original splat subsets, final transforms, camera matrices,
and PNG sequences in another animation/editor workflow. Generic mesh importers do
not necessarily understand Gaussian PLY attributes; use a compatible splat renderer
for `gaussians_*.ply`, not a normal point-cloud renderer.

### Coordinate math and why translations are used

Existing project transform:

```text
map_x = 1.43 * gaussian_x
map_y = 1.43 * gaussian_z
map_z = 1.43 * (1.1 - gaussian_y)
```

Values are read from Stage 4C config, not hardcoded into this module.

Only positions are translated in the original Gaussian frame during the explosion.
Scales, quaternion orientations and SH coefficients remain in their original axes.
No object rotation, normalization of size, or retraining is performed. Translating
an entire object preserves its orientation basis; appearance is reevaluated from
the current virtual camera by gsplat. The final exported gallery PLY therefore
needs only a translation of `x,y,z`, unlike a rotation or change of coordinate basis.

The map point cloud and Gaussian representation have different source geometry;
their visual transition is a fade/reveal, not a pointwise one-to-one reconstruction.
The last gallery keeps relative real/project scales rather than silently making
all objects the same size. Objects are arranged in a camera-facing spatial plane
using translations, with no collision/physical-motion claim.

Gaussian subsets can contain holes, halos, leaks, or overlapping part/assembly
hypotheses. Extraction limits do not disappear in a beautiful animation. The
2-sigma layout margin is a display heuristic, not an exact bound to infinite
Gaussian support. Large ghost splats may still require upstream cleanup.

## Adjustments (use a NEW plan directory)

- Larger objects / less visual clutter: `--objects 6` or `--objects 9`.
- Three columns: `--columns 3`.
- Wider gallery separation: `--gap 1.0`.
- Denser point-cloud background: `--env-points 350000`.
- Larger point dots: `--point-sigma 0.9`.
- Shorter movie: `--duration 12 --fps 24`.
- Lower-memory trial: `--width 960 --height 540 --env-points 100000 --object-points 8000`.
- Select objects explicitly: `--regions region_...,region_...`.

Do not edit a committed `plan.json` or `motion.npz` in place: resume checks protect
against mixing old frames with a new recipe. Original subsets are not reduced by
`--object-points`; that flag limits only the point-cloud display stage.

## Focused validation, not a claimed full scene test

- Python syntax/CLI checked.
- CPU fixture checked coordinate transforms and camera handedness, gallery bounds
  within the final view, transition endpoints, copied assets and JSON/NPZ contracts.
- Translation checks confirmed exported fixture Gaussian fields other than XYZ were
  unchanged. A minimal scalar-binary PLY fixture adapter was used for that check;
  the delivered program uses your installed `plyfile`.
- Software point-layout storyboard inspected for framing/text; this is NOT a
  Gaussian GPU rendering test and is not shipped as if it were the Valmet movie.
- Real FFmpeg encoding checked using synthetic PNG frames.
- No real Valmet Gaussian data or CUDA renderer was available here. Your ten-frame
  storyboard is the first end-to-end Gaussian rendering check.

## Primary references

- gsplat 1.5.3 rasterization/attributes:
  https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- FFmpeg image-sequence encoding:
  https://ffmpeg.org/ffmpeg-formats.html#image2-1

No model weights, fonts, proprietary scene data, reference video, or third-party
library source code is bundled in the ZIP. The input geometry is copied only when
you execute the tool on your own machine.
