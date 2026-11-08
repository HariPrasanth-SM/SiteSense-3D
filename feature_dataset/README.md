# Valmet Stage 3 — DINOv3 observation features

This is a new `feature_dataset/` module alongside the existing `segment_dataset/`,
`capture_dataset*/` and `avatar_demo/` folders. It consumes the original RGB images
and independent FastSAM masks; it does not load the Gaussian scene or change source
images, masks, calibration, QA decisions, maps, ROS, CUDA, or the working renderer.

**Scope:** archive visual descriptors for every existing mask, generate strict
keep-only candidate manifests, and review cross-station visual similarities.
**Not yet:** assign object identities/labels, merge masks, assign point ownership,
extract object PLYs, turn image queries into service goals, or perform text search.

## Dataset reviewed for this update

The supplied summary reports 2,000 planned frames, 1,999 segmented frames and
169,018 returned masks: 147,795 keep, 21,207 review, and 16 exclude. It reports
1,899 keep frames, 100 review frames, and one excluded frame. These are QA decisions,
not accuracy estimates or counts of distinct physical objects.

The supplied `frame_quality.csv` has 404 kept masks inside review frames. This new
stage archives those features too, but **does not use them in default association
candidates**: both the frame and mask must be `keep`. Before feature-validity checks,
the uploaded numbers imply 147,391 strict keep candidates, and 147,109 strict
geometry candidates. The live audit reads the actual local Stage 2 files and is
source of truth if policies/results have changed.

## Install

Close other GPU-heavy work. Save the ZIP in Downloads and extract into the project:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_dinov3_stage3.zip -d .
source .venv/bin/activate
```

No `pip install -U` is required. This module uses the existing torch, NumPy,
Pillow and Stage 2 dependencies. `run.sh` uses `.venv/bin/python` and deliberately
makes **no changes** to CUDA or `LD_LIBRARY_PATH`.

Keep `segment_dataset/` present: the reader reuses its exact source/mask contracts.

## 1. Audit source data — no model required

```bash
DATASET="outputs/datasets/valmet_views_soft200_v1"
SEG="outputs/segmentation/valmet_soft200_fastsam_v1"
FEAT="outputs/features/valmet_soft200_dinov3_b16_v1"

bash feature_dataset/run.sh audit \
  --dataset "$DATASET" --segmentation "$SEG" --out "$FEAT"
```

This validates the capture plan hash, segmentation run, active QA policy, mask
archive checksums, raw RGB and calibration checksums, observation row order, and
image/mask resolution. It rereads RGB/calibration/masks, not every saved depth
array: depth geometry comes from the previous QA stage and remains linked.

An excluded frame with no masks is recorded, not re-segmented. A readable zero-mask
frame is allowed. Incomplete segmentation of an otherwise eligible frame is an
error; finish Stage 2 first. All currently returned mask proposals are in scope,
including review/exclude masks. Empty masks get invalid/zero feature rows rather
than invented embeddings.

Read `audit_summary.json` for exact counts. Stop Stage 2/reselection before extracting
features; do not mutate source archives while this process is running.

## 2. Obtain the official model code and authorized weights

Recommended initial backbone: **DINOv3 ViT-B/16**, natural/web-image (LVD-1689M)
weights. It emits 768-dimensional patch/class tokens. The implementation also
supports ViT-S/16, ViT-S+/16 and ViT-L/16 when explicitly selected with matching
native checkpoints; begin with B/16 on the 8 GB laptop, batch size 16.

Clone code without installing the full training requirements or replacing PyTorch:

```bash
mkdir -p vendor models
if [ ! -f vendor/dinov3/dinov3/hub/backbones.py ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3.git vendor/dinov3
fi
```

Official weights require authorized access under Meta's DINOv3 model terms. Use
Meta's request form and select the native **ViT-B/16 LVD-1689M backbone .pth**:

https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

Official instructions explain that approved users receive model download URLs:
https://github.com/facebookresearch/dinov3#pretrained-models

If already downloaded, set WEIGHTS to that file. Typical original filename:
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.

Otherwise, after obtaining your authorized URL, download it locally. The following
asks for the URL interactively so it is not typed literally into shell history:

```bash
WEIGHTS="models/dinov3_vitb16_pretrain_lvd1689m.pth"
read -rsp "Paste your authorized ViT-B/16 download URL: " DINO_URL; echo
wget --https-only -O "$WEIGHTS.part" "$DINO_URL" && mv "$WEIGHTS.part" "$WEIGHTS"
unset DINO_URL
```

Do not share the signed URL or access token in logs/screenshots. A missing or
denied weight download cannot be solved by substituting an arbitrary checkpoint.
The archive contains no model weights or third-party DINO source code.

**Use a native `.pth` file, not Hugging Face `model.safetensors`.** They use different
weight naming/packaging; this adapter intentionally does not convert between them.
Existing weights are loaded with `torch.load(weights_only=True)` and strict key
matching into the official architecture. Only load trusted official checkpoints.
No feature extraction proceeds with randomly initialized or partially loaded weights.

### Verify the pretrained model interface

```bash
bash feature_dataset/run.sh check-model \
  --repo vendor/dinov3 \
  --weights "$WEIGHTS" \
  --arch dinov3_vitb16 \
  --device cuda:0 --precision bf16
```

Expected shapes:

```
Pretrained model forward PASS: (1, 256, 768) (1, 768)
```

This verifies loading/finite forward outputs for one 256x256 input, not retrieval
accuracy. The full pipeline uses only the official backbone module, not optional
training, depth, segmentation or text heads. Keep the checked-out code unchanged
once a feature run begins; its source hash and Git revision are saved.

## 3. Extract a representative pilot first

```bash
bash feature_dataset/run.sh extract \
  --dataset "$DATASET" --segmentation "$SEG" --out "$FEAT" \
  --repo vendor/dinov3 --weights "$WEIGHTS" \
  --arch dinov3_vitb16 --device cuda:0 --precision bf16 \
  --crop-size 256 --frame-side 640 --batch-size 16 \
  --pilot --pilot-frames 32
```

The pilot prioritizes kept frames across different stations, usually with two views
per selected station, and includes some review frames to test archival behavior.
All proposals in a selected frame are encoded, not only kept proposals. Review
observations remain excluded from the default QA-neighbor/candidate indexes.

### Descriptors saved for each mask

1. **`crop_mask_pool`**: original RGB region masked to ImageNet-mean background,
   padded bounding crop, aspect-preserving letterbox to 256x256, area-weighted
   average of foreground patch tokens. This is the default observation descriptor.
2. **`crop_cls`**: normalized class token from that same masked crop; retained as a
   separate optional descriptor rather than blindly averaged with others.
3. **`frame_mask_pool`**: mask-weighted patch tokens from a full-image forward pass,
   preserving context. Full image is not cropped and is limited to 640 pixels on
   its longest side, padded only as required for the /16 patch grid.

All final descriptors are L2-normalized, stored in float16, and should be promoted
to float32 and renormalized for similarity calculations. A token named `x_norm_*`
is layer-normalized by DINO; the tool additionally performs L2 normalization.
Mask coverage weights may be fractional; masks are never replaced by coarse labels.
Original image/mask coordinates and exact crop/resize/padding transforms are saved.
The full-image descriptors can share background/context near boundaries. A tiny
region may occupy less than one full-image patch; its zoomed crop can be helpful,
but upsampling cannot create visual information missing from the source image.

**Default dense cache:** full-image L2-normalized patch tokens are also saved once
per frame in `dense.npz` (30x40x768 for a 480x640 image). They can later be linked
to visible geometry using the stored preprocessing transform and original camera
metadata. Patch-grid centers are not exact physical surface points.

No global image embedding is reused as though it were each object's descriptor.
No N-by-N matrix of all 169k masks is constructed.

## 4. Inspect the feature pilot

```bash
xdg-open "$FEAT/qa/index.html"
xdg-open "$FEAT/qa/neighbors_001.jpg"
```

The QA sheet shows a query mask crop and appearance-similar candidates from other
camera stations. It uses only strict keep candidates. The query station is excluded
and at most one result per other station is shown. This reduces trivial duplicate-
view matches; it **does not prove that the same physical object was found**.
Similar motors at different places can still match. Pilot matches are drawn only
from completed feature frames, so absence of a good match is not proof of failure.

`qa/neighbors.json` retains the IDs, scores and partial geometry references for
review. These are diagnostic retrieval suggestions, NOT an association graph or
confirmed objects. No fixed similarity threshold is calibrated here.

Checkpoint to share: `summary.json` plus `qa/neighbors_001.jpg`. If model loading
fails, share the traceback and installed version information, never credential URLs.

## 5. Full extraction and resume

Use exactly the same recipe without `--pilot`:

```bash
set -o pipefail
bash feature_dataset/run.sh extract \
  --dataset "$DATASET" --segmentation "$SEG" --out "$FEAT" \
  --repo vendor/dinov3 --weights "$WEIGHTS" \
  --arch dinov3_vitb16 --device cuda:0 --precision bf16 \
  --crop-size 256 --frame-side 640 --batch-size 16 \
  2>&1 | tee -a "$FEAT/extract_console.log"
```

The model is loaded once. Each frame gets one full-image pass plus batches of its
masked crops. Arrays are written per frame, and GPU tensors from previous frames
are not kept. There are ~169k mask crops, not just 2k image forwards: use pilot timings
rather than the much faster Gaussian render timings to estimate full extraction.

`Ctrl+C` preserves completed frames. Rerun the command to verify and reuse complete
matching archives. A partial frame is recomputed. `--limit 100` processes at most
100 NEW frame archives, not 100 masks. One output lock prevents concurrent writers.

For CUDA OOM, close other GPU jobs and reduce `--batch-size 4`; that does not change
the feature recipe and is allowed when resuming. Small numerical differences from
batching/hardware can still occur. Changing precision/model/input resolution or
background/padding policy needs a NEW output folder; no incompatible feature mixing.

The full-image pass is batch size 1. If that itself exhausts memory, a smaller
`--frame-side` requires a new run. No automatic silent resolution/model fallback.

### Disk space

For your 169,018 masks with ViT-B/16:
- Three 768D float16 descriptor matrices: approximately 0.725 GiB before compression.
- Dense full-frame patch cache for 1,999 480x640 images: approximately 3.43 GiB.
- Total numeric arrays: approximately 4.16 GiB, before metadata/review files.

Compression and actual source sizes change the on-disk total. Reserve at least
7 GiB additional space. To omit the dense token cache, add `--no-dense` from the
FIRST extraction in a new folder; contextual pooled descriptors still remain.
The descriptor record `mask_indices` is always the original Stage 2 mask order.

## Saved files

```
outputs/features/valmet_soft200_dinov3_b16_v1/
  source.json
  inventory.jsonl
  audit_summary.json
  run.json                        # model/checkpoint/code hashes and recipe
  policies/<policy_hash>.json
  pilot_selection.json
  summary.json
  observations.jsonl              # ALL extracted masks, including review/exclude
  association_candidates.jsonl    # frame keep AND mask keep AND valid descriptor
  geometry_candidates.jsonl       # above plus Stage 2 geometry eligibility
  quarantined_observations.jsonl
  frames/<frame_id>/
    features.npz
    dense.npz                     # default; optional with --no-dense
    metadata.json                 # every crop transform + full frame transform
    complete.json                 # input binding/output checksums, written last
  qa/
    index.html
    neighbors_001.jpg
    neighbors.json
    summary.json
```

`features.npz` keys:

```
mask_indices
crop_mask_pool, crop_cls, frame_mask_pool
crop_valid, cls_valid, frame_valid
crop_patch_mass, crop_effective_patches
frame_patch_mass, frame_effective_patches
```

Do not assume all arrays are valid for an empty mask. Use validity flags. There is
no renumbering or winner-takes-all mask ownership. An identical or overlapping mask
is still its own observation. Each row joins to the original Stage 2
`observation_id`, raw mask archive, original RGB/depth, camera metadata, QA record,
and partial surface summary. Full source records remain in Stage 2; keep them.

`semantic_label` and `object_id` are deliberately null for every row. A future
multi-view system may assign one or several object hypotheses or leave them unset.
No point cloud is forced to receive a label.

### Loading descriptors

```python
import numpy as np

with np.load('PATH/frames/s012_v003/features.npz', allow_pickle=False) as d:
    vectors = d['crop_mask_pool'].astype(np.float32)
    valid = d['crop_valid']
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    # Row 5 corresponds to Stage 2 mask index 5, not a semantic object ID.
```

## QA/filter changes without re-encoding

Features are keyed by segmentation run + original mask index, not QA acceptance.
After a legitimate Stage 2 `reselect` (same RGB/masks), regenerate Stage 3 manifests:

```bash
bash feature_dataset/run.sh reindex --out "$FEAT"
```

This reads current Stage 2 quality decisions and rebuilds strict candidate lists
without loading DINO. `report --out "$FEAT"` likewise reindexes and rebuilds QA.
A policy file edited without completing Stage 2 reselection is refused.
Review masks never enter default association automatically through a high cosine score.

## Why geometry comes next, not inside the descriptor

DINO appearance is not calibrated instance identity. The later association stage
should consider camera proximity/orientation, common visible 3D support, depth and
occlusion consistency, mask extent/quality, and appearance; similar embeddings alone
are not sufficient for repeated industrial machines. It should retain ambiguous
edges instead of hard-merging everything. The present archive preserves these joins
without modifying the geometry. Expected Gaussian depth is still approximate.

DINOv3's ordinary visual backbone is not automatically a text-image retrieval model.
Text labels/captions or a text-aligned representation can be added later, separately.
No model fine-tuning or training is performed in this stage.

## Tests and validation

```bash
source .venv/bin/activate
python -m unittest discover -s feature_dataset/tests -v
```

The integration fixtures use the previously shipped Stage 2 tests, so retain that
folder. See VALIDATION.md. Local CPU tests use a deterministic token stub and do NOT
establish learned DINO retrieval accuracy or GPU memory/throughput. Actual native
pretrained loading and GPU inference must be checked on your laptop.

## Primary sources

- Official DINOv3 repo/model and access instructions:
  https://github.com/facebookresearch/dinov3
- ViT backbone API and dimensions:
  https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py
- `forward_features`, class/patch token fields:
  https://github.com/facebookresearch/dinov3/blob/main/dinov3/models/vision_transformer.py
- Official pretrained-weight access:
  https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

Crop masking, pooling, strict QA selection and archive design are this prototype's
engineering choices, not a claim of an official DINOv3 instance-association method.
