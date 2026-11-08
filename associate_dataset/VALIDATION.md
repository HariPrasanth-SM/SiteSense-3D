# Validation performed for Stage 4A

48 Stage 4A tests passed. All 46 existing Stage 3 tests and 41 Stage 2 tests also
passed unchanged. CPU environment: Python 3.13, NumPy 2.3.5, SciPy 1.17.0,
Pillow 12.3.0, OpenCV 4.13.0. Code uses Python 3.10-compatible syntax.

Checks include pixel-center round trips; rotated/translated cameras; same physical
surface under translated viewpoints; visually identical but spatially disjoint
regions; occlusion; in-front depth conflict; duplicate-pixel projection filtering;
nested/overlapping masks; strict frame+mask keep eligibility; partial-support
classification; source/archive corruption; stale policy and incomplete feature
refusal; bounded-memory catalog shape; input file preservation; deterministic
query resume; static HTML/JPEG reports; and binary XYZRGB PLY serialization.

Integration inputs were produced through the actual earlier Stage 2 and Stage 3
archive writers, using their synthetic scene/mask and deterministic feature stubs.
No pretrained DINO/real Valmet depth inference was run. No CUDA or Gaussian PLY was
available here. These tests establish implementation/file-contract behavior, not
instance-retrieval accuracy or throughput on the user's 147,109 observations.

A generated contact sheet was visually inspected for layout. Source images and
geometry in that validation fixture are artificial and are not distributed as
Valmet results. Current real-world thresholds remain starting heuristics.
