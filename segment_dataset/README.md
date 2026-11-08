# Valmet Stage 2 — FastSAM mask archive + reversible RGB/depth quality selection

This is an additive local module `segment_dataset/` for the completed calibrated
Gaussian capture. It does not change `capture_dataset*`, `camera_spawn_soft`,
`avatar_demo`, the PLY, source images/depth/maps, CUDA environment, or ROS files.

**This step:** audit all frames, segment a representative pilot, retain every
returned FastSAM mask independently, compute per-mask quality/depth statistics,
and create default usable/review/exclude manifests. **Not yet:** DINOv3, label
search, cross-view matching, instance fusion, full-object PLYs, navigation updates.

## 1. Install and prerequisites

Save `valmet_segmentation_stage2.zip` in Downloads, then:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_segmentation_stage2.zip -d .
source .venv/bin/activate
python -c "from ultralytics import FastSAM; import torch, cv2, scipy; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use the existing working environment and pretrained **FastSAM-s.pt**. No model
training or package upgrade is required. The launcher uses `.venv/bin/python`
and does not change `LD_LIBRARY_PATH`, CUDA, `.bashrc`, ROS, or dependencies.

Close the avatar renderer, RViz, and any other GPU-heavy job first. The source
Gaussian PLY is **not loaded** during segmentation; this stage reads saved frames.
One image at a time is processed, and the FastSAM model is loaded once per run.

The default `--model FastSAM-s.pt` searches the current project, `models/`, then
`weights/`. If not found, pass its actual path. It deliberately does not silently
download a different checkpoint. To find your already-downloaded weights:

```bash
find . -maxdepth 3 -type f -name 'FastSAM*.pt'
```

Only load trusted PyTorch model files. No weights are included in this archive.

## 2. Define source and output

```bash
DATASET="outputs/datasets/valmet_views_soft200_v1"
SEG="outputs/segmentation/valmet_soft200_fastsam_v1"
```

The output must be outside the source capture directory. Each source frame must
have the standard capture files and valid `complete.json` checksums. The plan
hash and per-frame calibration/resolution are checked. A corrupt source is flagged
and not silently resized, modified or repaired.

Your 1952 new captures plus 48 earlier pilot frames make 2000 complete source
frames. This stage uses `plan.json`, not a filename glob or the progress denominator.

## 3. Audit all source frames first (CPU only)

```bash
bash segment_dataset/run.sh audit --dataset "$DATASET" --out "$SEG"
xdg-open "$SEG/qa/index.html"
```

This scans RGB, raw float `depth_m.npy`, alpha, camera metadata and source checksums.
The grayscale depth PNG is not used for calculations. It creates frame reports
without importing PyTorch or FastSAM.

### Initial quality decisions

The following are **tunable heuristics**, NOT validated accuracy measurements:

* Depth support: finite positive expected depth, alpha >= 0.5, and provisional
  distance 0.15–25 m. Alpha is rendering coverage, not confidence.
* Skip severely unusable frames by default: less than 15% usable depth, >=90% of
  covered pixels closer than 0.15 m, or essentially uniform RGB (gray std <1.5 and
  Laplacian variance <0.5). Corrupt data/calibration are always excluded.
* Review rather than automatically delete low detail, possible blur, partial
  coverage, and partially near-surface views. A smooth machine or wall can be
  textureless without being erroneous.
* Capture-stage review flags are carried into the reports; they do NOT automatically
  blacklist a frame. The earlier six review flags are not proof of six bad views.

Camera-inside-equipment views can still have alpha=1 and plausible depth. These
checks cannot detect every rendering artifact. Pilot review remains necessary.

The report calls statuses `keep`, `review`, `exclude`. Source files never change.
Audit-only output has no masks yet. `not_yet_segmented_eligible_frames` reports
what is still awaiting FastSAM.

## 4. Segment the representative pilot

```bash
bash segment_dataset/run.sh segment \
  --dataset "$DATASET" \
  --out "$SEG" \
  --model FastSAM-s.pt \
  --device 0 \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.9 \
  --max-det 300 \
  --pilot
```

Defaults are also 640 / 0.25 / 0.9 / 300. The pilot selects up to 48 views with
capture-pilot examples, quality-flagged views, spatial coverage, and low/mid/high
cameras. It does not take the first 48 neighboring images. Severely excluded
frames may be counted among those 48 but skipped, so fewer than 48 frames may
actually be segmented.

The segment command verifies/audits source frames again. It does not duplicate
or rerender them. It uses `retina_masks=True` and **asserts original resolution**.
A mask/depth resolution mismatch aborts with an error; no blind letterbox resize.

PIL images are passed as RGB. A raw RGB ndarray is not passed to the BGR ndarray
API. No prompt is applied: all mask proposals returned after the configured model
confidence/NMS are archived. Proposals suppressed by the model cannot be recovered
by later filtering, and masks are NOT guaranteed complete industrial objects.

Open the results:

```bash
xdg-open "$SEG/qa/index.html"
xdg-open "$SEG/qa/contact_masks_001.jpg"
```

Click a frame in a review page to see its original image, mask-ID overlay, depth
support image and per-mask table. Green outlines are kept, amber review, red
excluded. Overlay labels are raw mask indices (e.g. `005` means `m0005`). At most
100 masks are drawn for legibility; the archive/table retains all returned masks.

**Stop at this checkpoint.** Share `summary.json`, `frame_quality.csv`, and the
mask contact sheet. Add one or two full original images and per-frame overlay
examples when the filtering or segmentation looks wrong. No 370 MB PLY or whole
render dataset needs to be uploaded for this first review.

## 5. Mask checks and reversible selection

Each mask retains its original pixels, including overlaps and nested proposals.
The pipeline measures area, model score, edge truncation, connected components,
local RGB detail, usable-depth fraction, depth-jump fraction and visible-surface
statistics. Geometric support excludes invalid/near/far pixels, pixels at depth
jumps, and one pixel of the mask boundary. **This geometric subset does not replace
the raw segmentation mask.** Thin objects remain archived even if erosion removes
their eligible geometry.

Current defaults:

* Exclude <32-pixel masks or masks with <20% usable depth; archive them nevertheless.
* Review small masks (<128 pixels), low-detail masks, very large masks (>85% image),
  fragmented masks, <70% usable depth, low score, and excessive internal depth jumps.
* Border-truncated masks receive a hint, not an automatic rejection.
* Possible floor/high-horizontal surfaces receive hints, not automatic deletion.
* A useful local mask can be accepted from a partially problematic `review` frame.
  A severely `exclude` frame does not produce default usable observations.
* `eligible_features` requires mask `keep`. `eligible_geometry` additionally needs
  >=32 supported interior pixels and >=50% geometric support.

The preview's floor/ceiling hints use approximate geometry, NOT semantic recognition.
FastSAM is class-agnostic here: no equipment names or object identities are claimed.
A high model score is not a guarantee of a correct mask or physical geometry.

Geometry metadata uses the exact saved K, camera-to-map transform and
(column+0.5, row+0.5) pixel centers. Up to 2048 valid interior samples provide
robust median/5th–95th-percentile summaries. They describe partial visible-surface
support, NOT a full object box, a service goal, or original Gaussian indices.
The expected-depth blending can put samples between surfaces; multi-view checks
remain a later step. Scale 1.43 is still provisional.

## 6. Complete the segmentation after inspecting the pilot

Run with the same model/settings, without `--pilot`:

```bash
set -o pipefail
bash segment_dataset/run.sh segment \
  --dataset "$DATASET" \
  --out "$SEG" \
  --model FastSAM-s.pt \
  --device 0 \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.9 \
  --max-det 300 \
  2>&1 | tee -a "$SEG/segment_console.log"
```

The model is loaded once and each original-size image is processed sequentially.
No full-dataset GPU batch and no Gaussian scene are loaded. Completed mask archives
are checksum-verified and skipped. `Ctrl+C` preserves completed frames. Repeating
the same command resumes; a write lock prevents two writers.

`--limit 100` bounds the number of NEW segmentation frames. Repeating this command
advances by up to 100 new eligible frames, rather than revisiting the first 100.

Checkpoints, inference settings, torch/torchvision/ultralytics versions and source
code are bound to `run.json`. Change a model/inference setting in a NEW output
folder. Quality-policy changes alone do not require another inference run.

If CUDA fails, first run the existing small PyTorch CUDA diagnostic. Do not blindly
reinstall torch or CUDA. This stage does not compile gsplat.

## 7. Storage and the mask format

Every processed frame has:

```text
outputs/segmentation/valmet_soft200_fastsam_v1/
  source.json                   # capture location, hash
  run.json                      # model/inference/environment identity
  quality_policy.json           # current thresholds (editable)
  policies/<hash>.json          # immutable saved threshold recipes
  frame_audit.jsonl
  frame_quality.csv
  review_frames.csv
  mask_quality.csv
  observations.jsonl            # all processed proposals with current decisions
  usable_observations.jsonl      # default future image-feature input
  geometry_observations.jsonl    # stricter future geometric fusion input
  review_observations.jsonl
  summary.json
  audit/<frame_id>.json
  frames/<frame_id>/
    masks.npz                   # all independent bit-packed masks, no flattening
    inference.json
    complete.json               # checksums, written last
  analysis/<policy_hash>/<frame_id>/
    observations.jsonl
    summary.json
    rgb.jpg                     # review thumbnail, NOT embedding input
    overlay.jpg                 # visualization, NOT embedding input
    depth_support.png           # shared filtered-depth support bitmap
  qa/
    index.html
    contact_masks_001.jpg
    frames_001.html
    ...
    <frame_id>.html
```

`masks.npz` stores `bits` (N,ceil(H*W/8)), `shape` = [N,H,W], `bitorder` = "little",
`scores`, and model `boxes`. It compresses the archive losslessly. Mask indices
remain stable for that inference run, even if later excluded. One pixel may belong
to multiple masks. No mask ownership competition, deduplication or object merging
is applied at this stage.

Unpack in Python:

```python
from segment_dataset.storage import unpack_masks
masks, scores, boxes = unpack_masks(".../frames/s012_v003/masks.npz")
# masks.shape == (N, original_height, original_width), dtype bool
# masks[5] is exactly raw proposal m0005, not an exclusive label map
```

Use `observation_id` (run ID + frame ID + mask ID) to join future features,
associations and object hypotheses. Source frames also retain their capture plan
hash. All image/depth/camera paths point to original source files; do not delete
or move the source capture while using this archive.

Disk use depends on proposal count. For illustration, 50 masks per image across
2000 640x480 images would be about 3.58 GiB bit-packed BEFORE compression, plus
metadata/review images. With many more proposals, storage grows. Leave several
GiB free (8 GiB or more is a practical starting reserve), and inspect pilot output
size rather than assuming a fixed final size. Old policy versions are retained
and can add more space. `summary.json` reports actual raw archive bytes.

## 8. Change filtering without rerunning FastSAM

Edit a copy of the generated `quality_policy.json`; all fields are documented by
`segment_dataset/quality.py`. Example of an alternative partial policy file:

```json
{
  "blur_lap_max": 8.0,
  "blur_gradient_max": 4.0,
  "small_mask_area": 96
}
```

Unspecified fields in a newly supplied policy JSON use the built-in defaults.
To preserve all current settings, copy the full generated policy and edit it.
Run:

```bash
bash segment_dataset/run.sh reselect \
  --dataset "$DATASET" --out "$SEG" \
  --policy path/to/your_full_or_partial_policy.json
```

This verifies inputs and recomputes mask QA/geometry statistics from saved masks;
it does not import torch or run FastSAM. It creates a new `analysis/<policy_hash>/`
and rebuilds current manifests. Raw mask archives are unchanged. No discarded
source images are deleted.

If a newly relaxed policy enables a frame that was skipped before inference, run
the ordinary `segment` command again after reselection; only previously unsegmented
eligible frames require inference. `--segment-excluded` can deliberately archive
proposals for readable severely flagged frames, but they remain excluded from
default downstream lists under the policy. It never bypasses integrity errors.

To rebuild only the static report with unchanged policy:

```bash
bash segment_dataset/run.sh report --dataset "$DATASET" --out "$SEG"
```

Do not edit committed capture plans to fix quality issues. Exclusion is implemented
through this separate stage, not by deleting or modifying source frames.

## 9. Export a single original mask/crop for inspection

Choose a processed frame and its mask index from the overlay/table:

```bash
bash segment_dataset/run.sh export-mask \
  --out "$SEG" --frame s012_v003 --mask 5 \
  --dest outputs/mask_examples/s012_v003_m0005
```

This saves a full-resolution binary mask, RGBA object view and bounding-box crop.
Example IDs are illustrative; use an actual processed frame/mask from your report.
It does not change the model archive or pretend the crop is a new full object.

## 10. Tests and scope

```bash
source .venv/bin/activate
python -m unittest discover -s segment_dataset/tests -v
```

CPU tests use synthetic capture files and a fake predictor to test orchestration.
They exercise corruption detection, exact/overlapping masks, empty proposals,
original resolution, partial-view salvage, expected-Z back-projection with half-
pixel centers, source preservation, pilot selection, resume, filtering-only reruns,
HTML reports and mask export. They do NOT evaluate FastSAM accuracy or real CUDA
inference. See VALIDATION.md for the exact tested scope/results.

## Sources used to verify API conventions

- FastSAM API and single-class proposal behavior:
  https://docs.ultralytics.com/models/fast-sam/
- Predict inputs, original-resolution retina masks, model thresholds:
  https://docs.ultralytics.com/modes/predict/
- FastSAM postprocessing/prompting:
  https://docs.ultralytics.com/reference/models/fastsam/predict/
- gsplat 1.5.3 expected-depth convention:
  https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html

Thresholds in this code are our initial heuristics, not thresholds asserted by
those projects. Offline source inspection cannot prove the industrial scan is
geometrically accurate; multi-view validation is still needed.
