# Valmet — hand-picked two-row rotating showcase

Additive module: `semantic_showcase/`. Keep your existing, background-fixed
`semantic_animation/`, `avatar_demo/`, and successful animation output.

## Result

The default result is **25 seconds total**, with the same resolution and FPS as
its base animation. It copies the existing successful PNG frames rather than
rerendering the whole room or the 115-region reveal. It then renders a new finale:

1. Begin at exactly the final all-region gallery camera and region positions.
2. Your selected regions translate forward into two rows. The remaining regions
   fade away during this transition; they remain present in the original intro.
3. The camera targets the first two columns (four regions), pans left to right,
   and the selected regions turn gently about their own centers.
4. The camera pulls back to show the complete selected two-row collection.

For an 18-second base at 24 FPS, the result is 432 copied intro frames + 168 new
frames = 600 frames = exactly 25 seconds. The code reads actual source timing;
it does not assume the earlier video is exactly 17 or 18 seconds.

Default formation and pullback each take about 1.4 seconds; the remaining finale
is the horizontal pan. Rotation totals **35 degrees**, alternating direction by
row. This is intentionally not a complete spin, because back surfaces may be
missing in the extracted subsets. No missing surfaces are generated.

Regions remain hypotheses, not verified complete assets. This is presentation
motion, not physics, manipulation, navigation or relighting.

## Install

Save `valmet_semantic_showcase.zip` in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_semantic_showcase.zip -d .
source .venv/bin/activate
```

No package installation or CUDA/driver change is required. The launcher uses the
existing `.venv/bin/python` and does not change library paths. FFmpeg from your
previous animation is reused for encoding. Original code and data are unchanged.

## 1. Choose the successful base animation directory

```bash
BASE="outputs/animations/valmet_reveal_all_v2"
PICK="outputs/animations/showcase_selection.json"
SHOW="outputs/animations/valmet_reveal_showcase25_v1"
```

Use the actual folder you successfully rendered if its name differs. It must
contain `plan.json`, `motion.npz`, `assets/`, `frames/`, `frames_clean/`,
`frame_metadata/`, and `render_environment.json`. The MP4 alone is not enough:
we need the saved final poses, assets, and source PNG sequence.

```bash
ls "$BASE/plan.json" "$BASE/motion.npz" "$BASE/render_environment.json"
```

## 2. Inspect and select 10–20 regions

```bash
bash semantic_showcase/run.sh pick \
  --base "$BASE" \
  --out "$PICK" \
  --port 8766 \
  --open-browser
```

Open `http://127.0.0.1:8766` if necessary. The picker binds only to the local
machine. It uses no web services, CDNs, model calls or CUDA rendering.

### Selecting

- The left panel shows every region already included in your successful movie.
- Click a card to inspect its cleaned point cloud on the right. Drag to orbit;
  use the mouse wheel to zoom. This lightweight preview displays points, not splats.
- Select supporting **masked crop**, **context crop**, or **full-image overlay**
  from the evidence menus. Frame ID, mask index, and station ID remain visible.
- Use **Add to finale** or **Add/remove selected region**.
- Filter by R-number, label or exact region ID; **Copy region ID** is also provided.
- Prefer coherent equipment/parts with useful source views over huge wall fragments
  or duplicates. Selection is entirely yours; there is no new automatic filtering.
- **Save selected IDs** writes directly to `$PICK` and shows the actual path.
  Existing selections are backed up automatically before a new save.
- **Download selection JSON** is an optional browser-download alternative.

The picker reads the copied asset evidence inside BASE, so old source captions
are the snapshot from your base movie. New edits to a semantic scene's labels
are not silently substituted into the already-prepared animation.

### Order matters: paired columns

```text
Top row:       1   3   5   7   9 ...
Bottom row:    2   4   6   8  10 ...
               --------
                first four close-up
```

Use Up/Down in the selection list to reorder. Put four strong examples first.
The camera targets two columns at a time; depending on real aspect ratios,
neighboring columns can also enter the frame. All selected objects retain their
relative original size by default. A very large region can force a wider camera.

The picker recommends 10–20 but permits 1–40 for small tests. For this short video,
12–16 is a practical first selection. You do not need to send IDs back to the
assistant: the next command consumes the file directly.

For true original-splat inspection before choosing, you can also use your existing
`semantic_scene_plus` viewer at a different port and copy matching region IDs.
The new picker needs no additional GPU session.

Close the picker tab and press Ctrl+C in its terminal once the selection is saved.

## 3. Prepare the 25-second plan

```bash
bash semantic_showcase/run.sh prepare \
  --base "$BASE" \
  --selection "$PICK" \
  --out "$SHOW" \
  --duration 25 \
  --rotation-deg 35
```

The base animation is kept intact and all its regions remain in the intro. Only
the new finale uses your selected list. The plan records every frame's camera,
object center, rotation, opacity, and explicit transforms. Source asset hashes
are checked and bound to the new plan.

Inspect the order and evidence contact sheet:

```bash
xdg-open "$SHOW/showcase_layout.jpg"
```

This is a selection/layout sheet, NOT a rendered Gaussian storyboard. Its two rows
match the final order. The exact gallery geometry is shown by the rendered preview.

Always use a NEW `--out` for changes to selection, duration, layout or rotation.
Do not hand-edit committed plan/motion files. Old and new videos remain separate.

## 4. Render the storyboard first

Close other Gaussian GPU renderers, then:

```bash
bash semantic_showcase/run.sh render --out "$SHOW" --preview
xdg-open "$SHOW/storyboard.jpg"
```

The preview copies a few original frames and renders spaced finale frames.
Original Gaussian subsets are reused, not point sprites or neural re-generations.

Check:
- Final wide view transitions smoothly into the selected-object formation.
- First four selected regions are the first close-up.
- Objects are sufficiently large and remain legible across the pan.
- The limited rotation does not expose unacceptable holes or leaked geometry.
- Final pullback fits the selected collection.

The new renderer preserves the previously fixed packed-background behavior:
`backgrounds=None` in rasterization, followed by alpha-transmittance compositing.

## 5. Finish and encode

```bash
set -o pipefail
bash semantic_showcase/run.sh render --out "$SHOW" \
  2>&1 | tee -a "$SHOW/render_console.log"

bash semantic_showcase/run.sh encode --out "$SHOW" --both
xdg-open "$SHOW/semantic_showcase.mp4"
```

Outputs:
- `semantic_showcase.mp4`: existing intro captions + new showcase captions/labels.
- `semantic_showcase_clean.mp4`: same content, without text.

No source audio is copied; both movies are silent. Encoding uses H.264/yuv420p.
All expected frames and hashes must pass before encoding.

Rendering is offline: playback FPS is not a required live rasterization rate.
Stop with Ctrl+C and repeat the identical render command to resume. `--limit 30`
processes at most 30 NEW output frames (including copied intro frames). The preview
is the easiest way to reach the finale immediately.

The original prefix is copied as PNGs, without resizing, color conversion or a
second GPU pass. With default timing it is byte-identical to the original PNGs.
The final MP4 is a fresh encoding, so compressed bitstreams are not byte-identical.

## Longer close-ups without exceeding 25 seconds

If 20 selected regions pan too quickly, prepare a NEW output with:

```bash
bash semantic_showcase/run.sh prepare \
  --base "$BASE" --selection "$PICK" \
  --out outputs/animations/valmet_reveal_showcase25_longtail_v2 \
  --duration 25 --intro-duration 15 --rotation-deg 35
```

This explicitly shortens the old sequence to 15 seconds by selecting saved source
frames; it DOES change its playback speed. It gives approximately 10 seconds to
the finale without rerendering the intro. No optical-flow interpolation is used.

Other prepare options:
- `--duration 24`: 24 seconds total. Must leave enough time after the intro.
- `--formation 1.8`: longer movement into the two rows.
- `--pullback 1.8`: longer final zoom-out.
- `--rotation-deg 0`: disable spinning to compare geometry; still pans and pulls back.
- `--rotation-deg 60`: larger turn, more risk of exposing uncaptured backs.
- `--gap 0.8`: increase spacing in provisional map metres.
- `--forward-distance 3`: explicit shift toward the gallery camera in map metres.
- `--hero-scale 1.2`: OPTIONAL uniform 20% enlargement of all selected objects.
  This is recorded presentation scaling, not a change to the physical data. The
  default is 1.0. Relative object sizes remain unchanged even with this option.

No independent object-size normalization is performed. The camera's fitting
bounds include original Gaussian display margins and are conservative; outlier
splats can make an asset look small. Pick coherent regions and check the storyboard.

## Rotation correctness

The finale rotates each object about its saved original map-frame center, around
map +Z. The same proper rotation is converted to the original Y-down Gaussian
frame. It transforms:

- Gaussian means about the pivot, then translates to the showcase position.
- Quaternion orientation by left multiplication, rotating anisotropic covariance axes.
- View-dependent SH appearance by inverse-transforming the camera direction into
  the original object's local axes, evaluating the original coefficients, then
  rasterizing those RGB colors. Coefficients are not left in a mismatched world basis.

The displayed colors use gsplat's SH evaluation followed by +0.5 / minimum-zero
activation, matching the original rasterization convention. Point-center rotations
alone would be insufficient for a proper Gaussian rotation. No retraining occurs.

This rotates the captured radiance field; it does not synthesize physically new
lighting, reflections, shadows, missing back surfaces or hidden geometry.

## Saved work products

```text
outputs/animations/valmet_reveal_showcase25_v1/
  plan.json
  selection.json
  source_plan.json
  layout.json
  showcase_layout.jpg
  motion.npz
  camera_path.jsonl
  intro_source_indices.npy
  frames_clean/000000.png ...
  frames/000000.png ...
  frame_metadata/000000.json ...
  render_environment.json
  render_summary.json
  storyboard.jpg
  preview.html
  semantic_showcase.mp4
  semantic_showcase_clean.mp4
```

`motion.npz` stores tail timestamps, all region centers, camera-to-map matrices,
per-region yaw/opacity/scale, and selected original-map-to-presentation transforms.
`camera_path.jsonl` also saves each camera in original Gaussian coordinates.
`source_plan.json` identifies original assets, IDs, source centers and appearance.

This output REFERENCES the original BASE asset directory rather than duplicating
all 115 Gaussian files. Keep BASE, including its assets and original PNGs. For an
archival copy, keep both directories together with the selection JSON and code;
paths may need explicit adjustment after moving to a different machine. Source
geometry is never overwritten. No incorrectly rotated standalone SH PLY is exported.

Disk space: allocate roughly another copy of the existing two PNG sequences plus
the new finale and MP4 files. A few GiB is a sensible reserve, but actual PNG sizes
vary. A 512 MiB minimum-free check prevents starting on a nearly full disk.

## Focused validation

- Eight CPU checks: initial-frame continuity, two-row order, pivots and rotations,
  final camera framing, pan direction, quaternion covariance rotation, SH-local
  view-direction equivalence, and Gaussian/map coordinate conversion.
- End-to-end file/preview/resume/encode check with a synthetic CPU image provider:
  an 18-second fixture plus a 7-second tail encoded to exactly 25.000000 seconds.
- Picker catalog, point-cloud endpoint and save API checked with synthetic records;
  Python and JavaScript syntax checked.
- Real Valmet geometry and CUDA rasterization were NOT run in this environment.
  The storyboard on your laptop is the first true-Gaussian test of the new rotation.

Primary API references:
- https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- https://docs.gsplat.studio/versions/1.5.3/apis/utils.html
- https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/rendering.py
