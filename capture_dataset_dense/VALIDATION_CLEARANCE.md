# Camera spawn clearance update — validation

- 57 dense-tool tests passed under Xvfb (37 previous + 20 new).
- All 30 original Stage 1 capture tests passed unchanged.
- UI inspected at 1440x900 using an artificial room, approved gray corridor,
  and near-wall rejected pins/samples.
- Source hashes checked: capture_dataset/capture.py and common.py, avatar_demo/core.py,
  render_worker.py and run.sh were unchanged relative to the supplied archives.
- No actual Gaussian PLY, CUDA rendering, or user's generated map was available for
  validation. We did not derive geometry from the screenshot or claim real clearance.

The tests initially included an overstrict assertion that rotated maps choose
identical farthest-point station IDs. Floating-point tie breaking can choose
another equally spaced station. That test now verifies coordinate reversibility
and identical distance fields rather than exact station IDs across rotations.
Same-input/seed reproducibility is tested separately and passes.
