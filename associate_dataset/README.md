# Valmet Stage 4A — geometry-verified, seed-based observation association

This is an additive, CPU-only `associate_dataset/` module. It reuses the saved
DINOv3 descriptors, original FastSAM masks, rendered depth/alpha, and exact camera
poses. It does not run a model, use CUDA, load the Gaussian PLY, change the input
files, or modify your viewer/navigation code.

**This stage is a matching pilot, not full-scene semantic segmentation.**

## The completed Stage 3 inputs

Your supplied summary reports 169,018 archived mask descriptors from 1,999
segmented frames; 147,391 strict appearance candidates and 147,109 strict geometry
candidates. There are 21,627 quarantined observations. These are observations of
regions in images, not numbers of unique physical assets. One source frame has no
mask archive. No labels or instance IDs have yet been assigned.

Stage 4A uses ONLY:

- `frame_status == keep`;
- `mask_status == keep`;
- valid crop features and Stage 2 feature eligibility;
- Stage 2 geometric eligibility and a usable visible-surface summary.

Review/excluded masks never become candidates through a high similarity score.
Descriptors for them remain untouched in Stage 3. The 282 appearance-only kept
observations without the stricter geometry eligibility are not used in this pilot.

## Install

Save `valmet_association_stage4a.zip` in Downloads:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_association_stage4a.zip -d .
source .venv/bin/activate
```

Keep `feature_dataset/` and `segment_dataset/` present; their source contracts and
depth-support functions are reused. No pip upgrade, new checkpoint, ROS launch, or
GPU installation is required. The launcher uses your existing `.venv/bin/python`.
It does not modify CUDA, `LD_LIBRARY_PATH`, `.bashrc`, or source data.

The default BLAS thread count is four to reduce CPU/memory contention. Set
`VALMET_BLAS_THREADS=2` for two threads. Source data must not be modified by another
capture/segmentation/reselection/extraction process while preparing or matching.

## 1. Prepare the keep-only search catalog

```bash
DATASET="outputs/datasets/valmet_views_soft200_v1"
SEG="outputs/segmentation/valmet_soft200_fastsam_v1"
FEAT="outputs/features/valmet_soft200_dinov3_b16_v1"
ASSOC="outputs/association/valmet_soft200_geometry_v1"

bash associate_dataset/run.sh prepare \
  --dataset "$DATASET" \
  --segmentation "$SEG" \
  --features "$FEAT" \
  --out "$ASSOC"
```

This checks source identity, Stage 2 policy, completed feature extraction, RGB /
camera / mask integrity, feature row alignment, and partial geometric summaries.
It copies only the kept `crop_mask_pool` vectors into a memory-mapped array, plus
compact observation metadata. It does NOT load the full image-patch feature cache,
DINO weights, FastSAM, or the Gaussian scene.

The expected count from your current files is **147,109**, but your actual local
files are checked. A mismatch is an error rather than silently changing selection.

Preparation needs at least 2 GiB free as a starting reserve for the index and pilot
reports. The 147,109 x 768 float16 crop matrix alone is about 216 MiB. Source data,
JSON records, caches, and reports add more. Memory use is bounded by chunked
similarity calculations and small geometry/frame caches; no N-by-N similarity
matrix is created.

If preparation is interrupted before the catalog completion marker, rerun it: the
CPU-only catalog is rebuilt. Once complete, repeating it verifies and reuses the
catalog. Source inputs are never overwritten. Use a new Stage 4 output after a
source/policy/feature-recipe change.

## 2. Run the geometry pilot

```bash
bash associate_dataset/run.sh pilot \
  --out "$ASSOC" \
  --name pilot_v1 \
  --queries 24 \
  --global-candidates 24 \
  --local-candidates 24
```

The pilot chooses up to 24 relatively usable masks from different stations,
spatially spread using their partial surface summaries. Very small/scene-sized and
floor-like masks are avoided as automatic SEEDS, but remain eligible CANDIDATES
when they passed the upstream keep/geometry policy. An automatic seed is not
asserted to be a complete machine.

For each seed it proposes up to 24 appearance-similar observations from other
stations, and up to 24 spatially overlapping candidate observations ranked by
appearance. At most two candidates per station are taken in each proposal branch;
the two branches are then deduplicated. These are bounded proposal searches, not
an exhaustive recall guarantee. Same-station comparisons are excluded.

### Evidence for each candidate pair

1. **Appearance:** normalized float32 comparison of saved crop features. Context
   descriptor similarity is also checked when valid; crop class-token similarity
   is recorded separately rather than averaged without justification.
2. **Coarse 3D location:** reject obviously separated partial-surface bounding
   regions, with padding. These summaries are incomplete, so this gate is a tunable
   conservative-to-moderate heuristic, not proof of different objects.
3. **Raw-mask visible geometry:** reconstruct up to 2,048 points per observation
   from the ORIGINAL mask, raw metre-depth, alpha, and exact camera-to-map matrix.
   Use Stage 2's valid-depth, discontinuity, and one-pixel interior filtering.
4. **Surface agreement:** measure both directions of nearest-neighbour overlap,
   using a SciPy cKDTree. Full object extent and equal centroids are not assumed.
5. **Bidirectional reprojection:** project seed samples into the other frame and
   the other samples back into the seed frame. Test FOV, positive camera-Z,
   target-depth validity, mask support, and visibility.

Predicted points farther than the observed surface are **occluded**, not matched
and not contradicted. Points significantly in front of the observed depth are
potential contradictions. Consistent depth outside the candidate mask is reported
as outside-mask evidence. Duplicate projections to the same target pixel are not
counted multiple times. Only the closest projected sample is used at that pixel.

Camera position and orientation are used through the complete camera transforms,
FOV and visibility tests. Baseline and relative viewing angle are reported, not
used to assume that only similarly oriented cameras can view the same object.

### Four output statuses

| Status | Meaning |
|---|---|
| `supported` | Appearance AND two-way surface/reprojection evidence pass the current policy. Still a hypothesis, not confirmed instance identity. |
| `partial_overlap` | One-sided/partial agreement, e.g. a part versus a larger assembly. Saved separately and not fused into the preview group. |
| `inconclusive` | Too little visible testable overlap, occlusion, or inadequate evidence. Not a claim that the physical objects are different. |
| `rejected` | Fails the current spatial/appearance/consistency rules. This is rejection of this proposed edge, not deletion of a source mask. |

No connected-component clustering or transitive chain merging is performed. Two
seeds can share supporting observations or have overlapping point samples.
`semantic_label`, `object_id`, and `navigation_goal` remain null.

## 3. Review the pilot and stop at this checkpoint

```bash
xdg-open "$ASSOC/runs/pilot_v1/qa/index.html"
xdg-open "$ASSOC/runs/pilot_v1/qa/associations_001.jpg"
```

The contact sheet shows each query and representative matches. Open an individual
query page for ALL tested candidates, including high-appearance matches rejected
because they are in different places.

A pair page includes the numeric scores and, for selected candidates, a two-way
projection overlay on the original images:

- Green: matching depth inside the other mask;
- Blue: occluded by nearer geometry (neither support nor contradiction);
- Pink: projected geometry in front of the observed depth;
- Yellow: matching depth outside the other mask.

The source mask outline is gold. Multiple geometric metrics are retained instead
of presenting one invented "confidence probability".

Check for:
- similar equipment in different places being separated;
- actual same-region observations being supported from multiple positions;
- a valve versus its larger assembly remaining a partial edge;
- false support from blurred or broad mask regions;
- overly strict thresholds leaving obvious matches inconclusive.

Share `runs/pilot_v1/summary.json`, `qa/associations_001.jpg`, and one interesting
query's `result.json`. Full scene files and feature matrices are not needed for
this first review. Do not expand this into global clustering before inspecting
some actual pair evidence.

## 4. Inspect a specific observation (optional)

For example, the earlier feature sheet included `s044_v000 m0079`:

```bash
bash associate_dataset/run.sh inspect \
  --out "$ASSOC" \
  --name valve_inspection_v1 \
  --frame s044_v000 \
  --mask 79 \
  --global-candidates 24 \
  --local-candidates 24

xdg-open "$ASSOC/runs/valve_inspection_v1/qa/index.html"
```

Use a frame/mask pair from your actual report. If it is not keep+geometry eligible,
the tool refuses it; there is no implicit review-mask bypass. This is a query by
an existing observation ID, not arbitrary-image or natural-language query input.

## Diagnostic 3D point clouds

For each seed, the pilot writes:

- `seed_surface_map.ply`: sampled visible surface belonging to that mask;
- `supported_surface_preview_map.ply`: seed plus up to eight directly supported
  observations, no more than one representative per additional camera station;
- corresponding NPZ arrays retaining source catalog-row indices and RGB;
- `hypothesis.json`: membership, partial bounds, provenance, and null labels.

The preview does NOT merge by chaining. Partial-overlap/inconclusive/rejected masks
are excluded from that union. A group receives `tentative_seed_region` status only
with at least three preview camera positions separated by 0.10 m. Otherwise its
status is `insufficient_multiview_support` and the seed can still be inspected.

**These are diagnostic point samples, NOT final segmented object models.** They
come from expected rendered depth; they are not indices into the original Gaussian
PLY. A few views do not reconstruct a whole machine, and part/assembly ambiguity
can persist. The median and percentile box are partial-surface summaries, not an
accessible navigation endpoint or a calibrated full-object boundary.

Cloud coordinates use the existing provisional ROS/map frame in metres:

```
x_map = 1.43 * x_Gaussian
y_map = 1.43 * z_Gaussian
z_map = 1.43 * (1.1 - y_Gaussian)
```

Metre-depth is paired with rigid `camera_to_map`; scale is not applied twice.
Pixel centers remain `(column + 0.5, row + 0.5)`. The exact convention is validated
using the existing Stage 2 code. The floor and scale remain approximate.

## Initial thresholds (explicit heuristics)

See `associate_dataset/default_match_policy.json`:

| Setting | Default |
|---|---:|
| Crop cosine minimum | 0.80 |
| Context cosine minimum, when valid | 0.55 |
| Near-surface distance | 0.12 provisional m |
| Depth consistency | 0.10 m + 3% of observed camera-Z |
| Two-way surface overlap fraction | 0.25 in EACH direction |
| Reprojection support fraction | 0.55 in EACH direction |
| Minimum supporting target pixels | 16 in EACH direction |
| Minimum testable source fraction | 0.05 in EACH direction |
| Maximum front-depth-conflict fraction | 0.25 |
| Target-mask boundary tolerance | 1 pixel |
| Coarse partial-bounds separation gate | 0.50 m |
| Maximum samples per mask | 2,048 |

These are NOT tuned or validated on the Valmet scene. Geometry tolerance may be
too loose for tiny neighboring parts, or too strict for noisy expected depth and
disjoint visible sides. Strong support is deliberately two-way; a match can be
missed if the two views see different sides of an object. That is preferable to
silently merging two identical-looking nearby machines in this first test.

To revise the policy, copy the JSON, edit it, and use a NEW run name:

```bash
cp associate_dataset/default_match_policy.json outputs/association_policy_v2.json
# Edit the JSON, then:
bash associate_dataset/run.sh pilot \
  --out "$ASSOC" --name pilot_v2 --queries 24 \
  --policy outputs/association_policy_v2.json
```

The catalog and all models/features are reused. Do not relax every threshold at
once. Compare concrete supported/rejected projections before revising settings.

## Resume, input preservation, and output structure

Completed query results have input bindings and output checksums. Repeating the
same command verifies and reuses complete queries. An interrupted query is
recomputed; earlier completed queries survive. If source content changes, the
command refuses to silently combine results. Rerun `prepare` with a NEW output
when source masks, QA selection, calibration, or feature recipe legitimately change.

```
outputs/association/valmet_soft200_geometry_v1/
  source.json
  catalog.jsonl
  crop_vectors.npy                   # memory-mapped float16 kept crop vectors
  frame_bindings.json
  catalog_complete.json
  runs/
    pilot_v1/
      run.json                       # policy, queries, exact source/index/code IDs
      summary.json
      pairs.jsonl
      supported_edges.jsonl
      partial_edges.jsonl
      seed_hypotheses.jsonl
      queries/
        q000_sXXX_vYYY_mZZZZ/
          result.json                # all candidate metrics and input provenance
          projection_checks.npz
          seed_surface_map.npz
          seed_surface_map.ply
          supported_surface_preview_map.npz
          supported_surface_preview_map.ply
          hypothesis.json
      qa/
        index.html
        associations_001.jpg
        q000_sXXX_vYYY_mZZZZ/index.html
        ...
```

To rebuild the report without rerunning comparisons:

```bash
bash associate_dataset/run.sh report --out "$ASSOC" --name pilot_v1
```

`report` uses saved results; it is not a source regeneration step. Do not modify
source images while using the pilot reports. The program never overwrites any
input capture, mask, feature, viewer, map, CUDA setting, or ROS file.

## Tests and limitations

```bash
source .venv/bin/activate
python -m unittest discover -s associate_dataset/tests -v
```

48 tests passed in the development environment. Tests include actual pixel-center
projection/back-projection, translated cameras observing the same plane, repeated
appearance at different positions, occlusion vs contradiction, nested masks,
keep-only indexing, integrity checks, resume, reports, and source preservation.
All 46 earlier Stage 3 tests and 41 Stage 2 tests also passed unchanged.

Test inference uses the earlier deterministic token stub, NOT a pretrained DINO
run. The artificial-frame integration test is for file contracts, not measurement
of real object association accuracy. A generated report was visually inspected.
The user's actual depth/mask/feature files are not available in this environment;
only the supplied summaries and sheets were reviewed. Run the pilot locally.

## Relevant primary references

- gsplat 1.5.3 expected-depth definition:
  https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- SciPy nearest-neighbour queries used for sample comparisons:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query.html

The thresholds, seed selection, grouping, and verification logic are prototype
engineering choices, not a claim of a published or calibrated instance method.
Expected Gaussian depth can be correlated across views and inaccurate; geometric
self-consistency does not prove real-world accuracy.
