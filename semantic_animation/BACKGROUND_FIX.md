# Valmet animation fix: packed RGB background + explicit all-region selection

This is an update for the existing `semantic_animation/` module. It changes only
`movie.py`, `plan.py`, and `__main__.py`. Keep the existing `common.py`, `run.sh`,
`avatar_demo/`, and original dataset. No PyTorch, gsplat, CUDA, driver, or package
installation changes are required.

## Why the render failed

The supplied animation passed a background with shape (1,3), which is the public
single-camera convention. In gsplat 1.5.3, `rasterize_to_pixels()` derives
`image_dims` from `means2d.shape[:-2]`. In packed mode, means2d has shape (nnz,2),
so image_dims is empty. The background assertion therefore rejects (1,3).

The patch keeps `packed=True`, supplies `backgrounds=None`, and composites the
background after rendering:

```python
rgb = rendered[0] + (1.0 - alpha[0]) * background.reshape(1, 1, 3)
```

`rendered` is already alpha-weighted; do NOT multiply it by alpha again. This
workaround does not edit the installed gsplat package or substitute point rendering
for the original Gaussian rasterizer.

Primary references:
- https://raw.githubusercontent.com/nerfstudio-project/gsplat/v1.5.3/gsplat/cuda/_wrapper.py
- https://docs.gsplat.studio/versions/1.5.3/_modules/gsplat/rendering.html

## Install

Stop any animation render first.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
cp -a semantic_animation "semantic_animation_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/valmet_animation_background_fix.zip -d .
source .venv/bin/activate
```

This patch does not change source images, masks, descriptors, associations, scene
labels, point clouds, Gaussian subsets, source PLY, or existing animation plans.

## All regions: use an explicit switch and a NEW plan

Previously, omitting `--regions` used automatic selection with a 12-region budget;
it did not mean all regions. Raising --objects alone also left the structural,
size, and overlap filters active.

`--all-regions` now selects every prepared region with a nonempty original Gaussian
subset. It bypasses the count, structure, size and shared-Gaussian suppression
rules. Small (1–19 Gaussian) subsets are included too. Separate overlapping region
hypotheses remain separate, not merged into unique physical objects.

Manually rejected regions remain excluded unless `--include-rejected` is explicitly
added. Regions with no prepared/nonempty original subset cannot be rendered as
original splats. All omissions and their reasons are recorded in
`selection_report.json`. The terminal prints included versus source-region counts.

```bash
ENH="outputs/semantic_enhance/valmet_regions150_v1"
ANIM="outputs/animations/valmet_reveal_all_v2"

bash semantic_animation/run.sh prepare \
  --enhancement "$ENH" \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --out "$ANIM" \
  --all-regions \
  --duration 18 \
  --fps 24 \
  --width 1280 --height 720 \
  --object-points 5000 \
  --portable
```

Use your actual enhancement path. Use a different NEW output folder if this one
already contains a plan. --object-points 5000 limits only point-sprite display
samples per region. The original cleaned clouds and complete original Gaussian
subsets remain exported, and no region is dropped to meet that display cap.

Optional additions when preparing:
- `--include-rejected`: also include manually rejected regions (only with --all-regions).
- `--camera outputs/renders/view_A_camera.json`: choose a known opening camera.
- `--columns 12`: explicit gallery column count; default is automatic.

Check the actual count:

```bash
python - "$ANIM/plan.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print("Mode:",p.get("selection_mode"))
print("Regions:",p["region_count"])
print("Frames:",p["frame_count"])
print("Selected subset Gaussians:",sum(r["gaussians"] for r in p["regions"]))
print("See selection_report.json for skipped source regions.")
PY
```

The sum includes any original Gaussian duplicated across independent hypotheses.
Many regions will appear small in a single final gallery; all selected regions
remain included. Existing largest-object slot sizing is unchanged.

## Test the exact failure first

```bash
bash semantic_animation/run.sh render --out "$ANIM" --limit 1
```

Expected artifact after success: `$ANIM/frames/000000.png`. This tests only the
opening room stage, not the later larger all-region bank.

Then test the full timeline at ten representative frames:

```bash
bash semantic_animation/run.sh render --out "$ANIM" --preview
xdg-open "$ANIM/storyboard.jpg"
```

Full render and encoding:

```bash
set -o pipefail
bash semantic_animation/run.sh render --out "$ANIM" \
  2>&1 | tee -a "$ANIM/render_console.log"
bash semantic_animation/run.sh encode --out "$ANIM" --both
```

Outputs remain `semantic_reveal.mp4` and `semantic_reveal_clean.mp4`.

## GPU memory changes

The renderer now allocates one bank directly instead of retaining all source
subsets and a second concatenated copy on the GPU. It loads representation types
only when active on the timeline: environment points, region points, original
region splats. The room is still released after its opening phase.

This reduces avoidable duplication. It is NOT a guarantee that every possible
all-region set fits 8 GB. The log reports tensor-size estimates and free GPU memory;
rasterization needs additional buffers. Nothing is silently decimated or omitted.

Close other GPU-heavy applications. If a later frame raises CUDA OOM, keep the
traceback and the active-bank log. Lowering resolution in a NEW plan can reduce
rasterizer-buffer use but does not reduce the full Gaussian parameter bank.

## Retrying the old failed plan

The old plan retains its old region selection; updating code cannot add regions
to an already-saved animation. For all regions, prepare the NEW plan above.

If you only want to test the existing plan, rerun render. When a failed attempt
left `render_environment.json` but NO completed frame metadata, the patch preserves
that old binding as `render_environment.failed_<timestamp>.json` and records the
new recipe. If completed frames exist under a different code/runtime recipe, it
refuses to mix them: use a new output directory. Do not manually delete checksum
records to force mixed frames.

## Focused validation

Eight CPU tests passed: packed-background assertion reproduction, alpha compositing
at alpha 0/0.5/1, selection of 150 regions without heuristic suppression, unchanged
automatic budget, empty-selection error, active-layer preallocation, empty-failed-run
recovery and refusal to mix completed frames from a different recipe. Python syntax
and the command-line parser were checked as well.

GPU rasterization and the actual Valmet data were NOT run here. GPU calls were
mocked in these focused tests; they establish the argument/compositing changes,
not end-to-end rendering accuracy or laptop memory requirements.
