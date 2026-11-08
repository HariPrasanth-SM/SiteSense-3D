# Validation performed in this session

## Tested

- **41 CPU unit/integration tests passed**, run with `python -m unittest discover -s segment_dataset/tests -v`.
- The integration tests use synthetic camera/RGB/depth files and an explicitly fake predictor.
- A separate compatibility check used the actual previously supplied `capture_dataset.common.save_frame` writer and then loaded its output with this Stage-2 reader. Hashes, expected-Z conventions, camera matrices and half-pixel conventions matched.
- Module compilation and CLI help were checked.
- A synthetic mask contact sheet was generated and visually inspected.
- The static HTML pages were generated and exercised by tests. A browser screenshot attempt did not complete in this container, so browser-layout verification is not claimed.

The tests cover exact bit-packed round trips (including overlapping/nested/zero
masks), wrong-resolution failures, source checksums, plan identity, original-file
preservation, blank/near/invalid/low-coverage frames, local-mask salvage from
partial views, thin objects, depth edges, camera rigidity and pixel centers,
pilot selection, count limits, checkpoint-config mismatch, resume, raw archive
corruption, QA-only reselection and individual mask export.

## Not tested / unavailable here

- Real FastSAM inference, GPU execution, your actual 2000-frame dataset, and the actual PLY.
- This container has no installed Ultralytics or FastSAM weights. An isolated dependency-download attempt was unavailable because the environment could not resolve the package index; no successful model-inference test is implied.
- Actual proposal quality, thresholds, image rejection precision/recall and physical scan geometry.

Use the existing FastSAM installation/weights that already worked on your laptop.
Run the 48-view pilot before processing the complete source dataset. No actual
user source frame was excluded here; only the local program will make those
heuristic decisions and preserve its evidence.
