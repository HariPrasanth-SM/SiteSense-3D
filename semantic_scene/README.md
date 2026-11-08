# Valmet Stage 4B — selectable 3D regions and a semantic review layer

This adds `semantic_scene/` only. Existing `associate_dataset/`, `feature_dataset/`,
`segment_dataset/`, source captures/masks/depth/PLY, navigation, ROS and CUDA are unchanged.

## Purpose

1. Convert directly supported Stage 4A seed groups to denser, independent point clouds.
2. Overlay them and their 3D bounding frames on your existing map-frame environment cloud.
3. Review the original source crops/poses; isolate or reject incorrect regions.
4. Optionally obtain OpenCLIP category suggestions from several kept observations.
5. Save manually reviewed labels separately from automatic suggestions and geometry.

A region ID is a stable **hypothesis ID**, not a verified physical asset ID. Multiple
regions can describe the same item or overlapping part/assembly. Unassigned points
remain unlabelled. This is not an exclusive full-scene point-cloud segmentation.

The uploaded Stage 4A summary has 24 queries, 727 tested pairs (220 supported,
455 rejected, 23 inconclusive, 29 partial) and 19 tentative groups. Those counts
are not accuracy estimates. This tool starts from those groups; it does not silently
assign the other ~147k observations to objects or recover all boiler-room equipment.

## Install

Save `valmet_semantic_scene_stage4b.zip` in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_semantic_scene_stage4b.zip -d .
source .venv/bin/activate
```

Keep your existing upstream folders present. Build/view use the existing NumPy,
SciPy and Pillow installations plus Python's standard library. **No new dependency,
renderer, ROS process, model or CUDA configuration is required for build/view.**

## 1. Build a scene from the current pilot

```bash
ASSOC="outputs/association/valmet_soft200_geometry_v1"
SCENE="outputs/semantic_scene/valmet_pilot_v1"

bash semantic_scene/run.sh build \
  --association "$ASSOC" \
  --run pilot_v1 \
  --out "$SCENE"
```

Use the actual association run name if yours differs. This reads the existing
`runs/pilot_v1/queries/*/result.json`, not the uploaded contact sheet.

Defaults:

| Setting | Default |
|---|---|
| Minimum distinct camera positions | 3, separated by >= 0.10 provisional m |
| Observations used per region | Seed plus up to 8 directly supported observations, one per station |
| Samples per observation | Up to 12,000 valid mask-interior depth samples |
| Voxel resolution | 0.02 provisional m |
| Selected masks | Strict frame keep + mask keep + geometry eligible |
| Partial/inconclusive/rejected associations | Not fused |

We re-back-project masks using the existing calibration and Stage 2 depth support
rules. DINOv3 and FastSAM inference are not repeated. The original 2,048-point pair
checks are not repeated with the denser set, so denser samples do not themselves
increase confidence in the region relationship.

Within each voxel, different observations contribute equally rather than the
densest view dominating. Each voxel retains its contributing observation list.
`station_count >= 2` identifies voxels supported by multiple stations **at the chosen
voxel resolution**; it is not a probability of correctness. Expected-depth offsets
and voxel boundaries can leave a true surface with only one station per voxel.

Groups remain independent, with no transitive merging, nearest-object assignment,
forced label ownership, or erasure of overlapping samples. The box encloses the
observed samples, not a hidden or complete equipment volume.

The builder records source/query hashes, rejects incompatible output recipes and
verifies/reuses completed region archives on rerun. It also rechecks source frame
integrity. A new recipe needs a new output directory. Stop upstream reselection or
association writes before building.

Optional checks:

```bash
bash semantic_scene/run.sh check --scene "$SCENE"
```

## 2. Open the local interactive 3D viewer

Use your already-exported map-coordinate environment NPZ (NOT the Gaussian PLY):

```bash
bash semantic_scene/run.sh view \
  --scene "$SCENE" \
  --environment outputs/ros/valmet_colored_cloud.npz \
  --port 8765 \
  --open-browser
```

Open `http://127.0.0.1:8765` if the browser does not launch. Keep the terminal open.
No map server, ROS, RViz or avatar process is needed. The server binds only to
127.0.0.1. There is no public sharing link, external CDN, telemetry or scene upload.
The HTML/JavaScript ships inside the package.

The environment NPZ must have `xyz` and `rgb` arrays in the existing ROS/map frame.
It is NOT transformed a second time. The default cap is 200,000 environment points
and 15,000 display points per region; complete exported PLY/NPZ clouds are retained.
A higher-resolution source can also be passed and downsampled in memory:

```bash
bash semantic_scene/run.sh view \
  --scene "$SCENE" \
  --environment outputs/ros/valmet_colored_cloud_full.npz \
  --max-env-points 350000 \
  --port 8765
```

Omit `--environment` to inspect only the extracted regions. If the port is busy,
close the old viewer or select another port and use its printed address.

### Controls

- Drag to orbit; Shift-drag or right-drag to pan; mouse wheel to zoom.
- Select a region from the list or click its bounding box. Large overlapping boxes
  can intercept clicks, so the list is the reliable alternative.
- **Focus selected** centers the view on that region; **Whole scene** restores context.
- **Isolate selected region** hides other hypothesis overlays, not the environment.
  Uncheck **Environment context** to view only the selected point cloud.
- Switch **Distinct region colors / Original RGB**.
- Switch **All observed samples / Voxels supported by >=2 stations**. The latter can
  be sparse and does not mean the other points are false.
- **Hide environment above Z** is a display-only ceiling cutoff. It does not edit
  geometry or remove high equipment samples from region archives.
- Each crop opens its original image with a mask outline. The cyan frustum is the
  source observation camera, not the current browser camera or a navigation goal.
- **Save screenshot** exports the rendered canvas with text labels.
- Filter by a region ID, reviewed label or suggested category. This is string
  filtering over metadata, NOT arbitrary DINO text retrieval.

Browser WebGL is used when available. If unavailable, a lower-density software
canvas preview is used automatically; its painter-style occlusion is approximate.
For the final demo, enable browser hardware acceleration. Source clouds remain full
resolution regardless of the display cap or fallback.

No pretrained model runs while viewing. No real-time Gaussian splat view is embedded:
this is a colored point-cloud semantic/spatial view, complementary to your existing
photorealistic avatar renderer.

## 3. Give regions semantic names

There are two routes; neither changes grouping automatically.

### A. Immediate manual labels (no model installation)

Inspect several source crops and the 3D region. Enter e.g. `door`, `cylindrical tank`,
`pipe assembly`, or leave the label blank. Select the kind and review decision, then
press **Save review**. An accepted region can still be unlabelled.

- Blank label: unknown.
- Accepted region: suitable for your demo, NOT measured ground-truth geometry.
- Rejected: hidden by default, but preserved; toggle **Show manually rejected regions**.
- Structural regions can be hidden separately after labels/kinds are provided.

Changes go to `review.json` and `review_history.jsonl`, not `region.json` or source
mask/association data. Concurrent edits use revision checks. A human name is a
separate record from the original automatic label proposal.

### B. Optional automatic category suggestions with OpenCLIP

DINOv3 descriptors are used upstream for appearance-based association. This module
adds a separate text-image model solely to suggest common categories. We do not
rerun DINOv3 or attempt to compare DINO vectors to unrelated CLIP text vectors.

Freeze key working-package versions before adding the OPTIONAL dependency:

```bash
source .venv/bin/activate
mkdir -p outputs
python - <<'PY'
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
names=['torch','torchvision','numpy','scipy','Pillow','gsplat','ultralytics']
lines=[]
for name in names:
    try: lines.append(f'{name}=={version(name)}')
    except PackageNotFoundError: pass
p=Path('outputs/semantic_scene_constraints.txt')
p.write_text('\n'.join(lines)+'\n')
print(p.read_text())
PY
python -m pip install 'open_clip_torch>=2.24,<4' \
  -c outputs/semantic_scene_constraints.txt
```

The constraints prevent this installation from replacing those pinned working
packages. If dependency resolution fails, do not bypass constraints or upgrade
CUDA/PyTorch; use manual labels for the current demo and retain the error.

Run with the official documented model/checkpoint pair:

```bash
bash semantic_scene/run.sh suggest-labels \
  --scene "$SCENE" \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --device cuda:0
```

The first model load may download its checkpoint. Downloading weights requires
network access, but inference is local: images are not uploaded. An existing trusted
native OpenCLIP checkpoint path can be supplied to `--pretrained`. Do not use the
DINOv3 .pth here; it is a different architecture/feature space.

Run with `--device cpu` to avoid CUDA for this optional step. This does not change the
avatar renderer. No model training is performed.

For each region, up to 3 source views are sampled from its directly supported
members. Both the masked crop and matched contextual crop are encoded; their
normalized vectors are combined at 0.7/0.3. Two text prompts per category are encoded
and averaged. Multi-view category scores are averaged. This is our uncalibrated
engineering heuristic, NOT a guaranteed industrial classifier.

Defaults require:
- At least 2 views.
- Highest mean cosine score >= 0.22.
- Top-vs-second mean margin >= 0.025.
- Agreement on the top class in >= 67% of views.
- Winning category not one of the explicit unknown categories.

Otherwise `suggested_label` remains null. Scores and top candidates are still
retained. Similarity is not a probability. For three views, the 0.67 rule effectively
requires all three; use a new explicit policy if changing this behavior.

Vocabulary lives at `semantic_scene/categories.json`. It includes wall/floor/ceiling,
door, cabinet, tank, pipe, valve, gauge, motor, pump, assembly and unknown options.
Custom JSON must provide unique `label`, `kind`, `description` for every category.
Use `--categories your_vocabulary.json` for a recorded alternative. Selecting a new
vocabulary requires rescoring these region crops, not rerunning segmentation or DINO.

Suggestions go to `semantic_suggestions.json` and immutable `label_runs/<hash>/`.
The loaded model parameter hash, package versions, prompts, numerical per-view
scores and selected observations are saved. Reviewed labels are never overwritten.
Refresh the browser after generating suggestions. `Copy suggested label` only copies
text to the editing field; review it and click Save to accept it.

**Do not identify maintenance condition, asset serial ID, or exact machinery subtype
from these categories.** A red tank cannot be assumed faulty. Workbook mapping and
service diagnostics remain separate, later steps.

## 4. Individual clouds and provenance

```
outputs/semantic_scene/valmet_pilot_v1/
  build.json
  scene.json
  summary.json
  regions.csv
  review.json                     # created by manual edits
  review_history.jsonl
  semantic_suggestions.json        # optional
  label_runs/<recipe_hash>/
    suggestions.json
    clip_vectors.npz
  regions/region_<stable_hash>/
    region.json
    samples.npz                   # xyz/rgb/source_uv/observation_index/station_index
    surface.npz                   # fused xyz/rgb/counts + many-to-many provenance
    surface.ply                   # all fused visible samples
    multiview_voxels.ply           # only voxels with >=2 contributing stations
    descriptors.npz               # member DINO descriptors and aggregate
    complete.json
    views/
      v00_masked.png
      v00_context.png
      v00_overview.jpg
      ...
```

`surface.npz` contains `membership_offsets` and `membership_observation_indices` in
CSR form. For voxel k, slice observation IDs between offsets[k] and offsets[k+1].
Those local IDs index `region.json:views`. `samples.npz` keeps exact source integer
pixels; upstream backprojection uses column+0.5,row+0.5 and metric depth. No second
scene scaling is applied to these points. The full original depth remains in capture.

The normalized aggregate DINO descriptor is stored for later visual retrieval; this
version does not provide arbitrary-image search or send navigation goals.

## 5. Expand beyond the 24-query pilot after the first 3D check

The current pilot only yields a small number of region hypotheses. To increase
coverage without repeating capture/FastSAM/DINO:

```bash
bash associate_dataset/run.sh pilot \
  --out "$ASSOC" --name regions_150_v1 \
  --queries 150 --global-candidates 24 --local-candidates 24

bash semantic_scene/run.sh build \
  --association "$ASSOC" --run regions_150_v1 \
  --out outputs/semantic_scene/valmet_regions150_v1
```

The old association tool permits up to 200 seeds and at most one automatic seed per
station. More seeds do NOT guarantee that every object is found. Structural regions
or similar fragment seeds can dominate. Use the existing `inspect --frame ... --mask ...`
workflow to target a specific missed mask. Do not assume 150 queries produce 150
unique physical assets. Near-duplicate hypotheses remain independently inspectable.

For the competition, review and label a small set of coherent equipment regions
before presenting a large automatic inventory. We can add conservative duplicate
handling later while preserving part/assembly overlaps.

## Validation and limits

- 38 new CPU/API tests passed: voxel support/provenance, no exclusive ownership,
  category-abstention logic, source-preserving file build/resume, labeling, local HTTP
  routes, edit revision/token checks and point-cloud serialization.
- Original 48 Stage 4A, 46 Stage 3 and 41 Stage 2 tests passed unchanged.
- A headless Chromium UI test exercised selection, focus, editing/saving a label,
  multi-view mode, isolation, search filtering and reset using a synthetic scene.
  This environment disables browser WebGL and direct browser loopback requests:
  the test used the actual software fallback and bridged browser requests to the
  local HTTP server. WebGL shaders passed syntax review but GPU rendering was not
  executed here. Test images are artificial, not Valmet extraction results.
- Integration uses the actual earlier archive writers and verifier. Its test fixture
  has coincident cameras, so a fixture-only distinct-view-count mock is used for the
  builder serialization check; separate geometric/pose tests test distinct counting.
- Pretrained OpenCLIP inference and your actual geometry dataset have NOT been run
  in this environment. Start with current pilot build/view, then optional labels.

Run tests:

```bash
python -m unittest discover -s semantic_scene/tests -v
```

## Sources for external APIs

- OpenCLIP official usage/model pair: https://github.com/mlfoundations/open_clip
- WebGL browser graphics: https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API
- Gaussian expected-depth meaning: https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html

No copies of third-party model weights, fonts, Valmet images, proprietary geometry,
or private source datasets are included in this code archive.
