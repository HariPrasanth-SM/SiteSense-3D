# Planned downstream contract — NOT implemented in Stage 1

The goal is a reversible observation-to-object database. A mask is a candidate
observation, not automatically a whole object. A cluster/box is a hypothesis, not
an identified industrial asset. Persist provenance and allow overlapping hypotheses.

## Relationships

```text
Dataset/scene version
  -> calibrated frame (frame_id, station_id, exact RGB/depth/alpha/K/camera matrices)
  -> mask observation (mask_id, frame_id, original-pixel mask)
  -> embedding record (mask_id, model/preprocessing/representation version)
  -> 3D visible-surface observation (mask_id, point provenance)
  -> association edge (mask_i, mask_j, geometric/appearance evidence)
  -> object hypothesis (object_id, many mask_ids, one or more point-set memberships)
  -> retrieval result and candidate service approach/observation poses
```

## Stage 2 — Mask archive

Keep every returned FastSAM proposal under a deterministic ID within a specific
segmentation run. Keep original image dimensions, area, box, proposal score,
model version, checkpoint hash and inference arguments. Preserve masks independently
(RLE or bit-packed arrays) rather than forcing one label per pixel. Overlapping and
nested masks are allowed. Export crops/overlays separately for inspection.

Use original-size masks (`retina_masks=True`) and assert mask/depth dimensions.
Do not blindly stretch a letterboxed mask to image dimensions. A changed mask
model or proposal threshold may require rerunning segmentation; a later size/height
filter need not. Keep a moderately broad base proposal archive, but no model can
save proposals it never produced.

Useful adjustable downstream filters, rather than permanent deletion:

- minimum/maximum pixel area or area fraction;
- image-edge truncation;
- valid-depth support ratio;
- 3-D extent, median height and proximity to floor/ceiling;
- outlier fraction and disconnected 3-D components;
- planar/background likelihood;
- observation count from distinct camera positions;
- geometric and visual association thresholds.

Small high-value gauges/valves should not all be removed by an aggressive size cutoff.

## Stage 3 — Features

Cache L2-normalized DINOv3 visual features with exact crop/mask/resize/pooling
metadata. Mask-pooled dense patch features and a crop descriptor can be separate
representations. A bank of view descriptors is useful for viewpoint-dependent
retrieval; do not necessarily average every observation into one feature.

Ordinary DINOv3 backbones provide visual embeddings, not automatically a compatible
text encoder. For label queries, add explicit labels/captions or a separate
text-image-aligned representation. Do not compare arbitrary text vectors with
DINOv3 embeddings or train a large text alignment model for this hackathon.

## Stage 4 — Geometry and association

For each mask, use calibrated expected depth to create visible-surface samples.
Retain source pixel/frame/mask information. The resulting point indices are NEW
surface-sample IDs, not automatically original Gaussian vertex IDs.

Use spatially nearby cameras / overlapping frusta and compatible 3-D observations
to propose matching pairs, then visual features and multi-view geometric checks.
Do not compare every mask with every other mask, or merge two physically distinct
motors merely because they look alike. Save edge scores and acceptance/rejection
reasons; rerunning associations should not require repeating rendering or features.

Nested masks (pump vs pump+pipe assembly) can be stored as separate hypotheses,
parent/part relations or ambiguous associations rather than forcibly merged.
Shared point memberships are explicitly allowed. Depth-edge artifacts and blending
can create false points; use depth continuity/multi-view agreement and retain raw
observations. One view only observes part of an object: boxes are candidate extents,
not guaranteed complete physical bounds.

Extracting original Gaussian subsets later is a DIFFERENT operation: needs mapping
to original stable PLY vertex indices with visibility/depth/contribution checks.
A depth sample is not a Gaussian ID, and nearest-center assignment alone is not a
robust semantic segmentation of the splats.

## Retrieval and navigation

Image query -> ranked observations -> object hypotheses -> 3-D location plus
source views. Label query needs a text-aware label/feature layer. Similar repeated
assets may remain ambiguous; return candidates instead of inventing identity.

Store separately:

1. Object location / candidate extent.
2. Camera observation poses that see it (possibly in gray capture-only space).
3. Reachable service approach poses validated on the navigation map.

Never reuse a synthetic observation pose in an unknown corridor as a navigation
goal just because rendering worked there. Point centers/asset boxes are also not
necessarily valid stopping points.

For visualization, keep the entire environment and overlay a selected object's
point set/box. Do not claim a movable, complete rigid object if the extraction is
only a partial visible surface.
