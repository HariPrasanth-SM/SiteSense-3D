# Validation record

## Executed in this environment

- 46 new CPU unit/integration tests passed.
- All 41 previously shipped Stage 2 tests passed unchanged.
- Python compileall and shell syntax validation passed.
- New integration tests generate captures using the original Stage 2 test fixture,
  create masks using its test predictor, and exercise the new source reader and
  feature pipeline using a deterministic algebraic token encoder.
- Tests cover aspect-ratio-preserving preprocessing, neutral backgrounds,
  crop pixel-edge transforms, area-based patch-mask weights, unit normalization,
  overlap preservation, zero/tiny masks, per-frame feature ordering, strict keep
  gating for both frame and mask, reference lineage, corruption rejection, resume,
  changed-recipe rejection, batch-size-only resume, optional dense output,
  late quality reselection without model inference, and source-file preservation.
- Review generator excludes each query's station and does not assign identities.

## Not executed here

- No official pretrained DINOv3 weights were available for a learned forward test.
- No NVIDIA GPU was available; CUDA inference, memory use and elapsed runtime are
  unmeasured. Model integration follows the inspected official native backbone API.
- The user's source images and bit-packed masks are not uploaded. The supplied
  summary/CSV/contact sheet were inspected, but real-dataset feature quality cannot
  be evaluated until the user runs the pilot.
- No association/instance fusion or object point-cloud extraction was implemented.

No masks, model weights, industrial source data, third-party model code, or fonts
are distributed in this package. The archive contains only the new module.
