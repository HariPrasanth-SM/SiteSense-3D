# Validation

Validated in this environment on synthetic inputs:

- 24 CPU tests: ROS trinary map interpretation, camera station sampling, spacing,
  deterministic IDs, explicitly approved gray pins, blocked occupied pins, exclusions,
  immutable plans and hashes, pilot sampling, camera/map coordinate consistency,
  raw-depth preservation, validity masks, checksum-based resumption, and contact sheets.
- 6 Tk UI tests under Xvfb at 1440x900: gray selection opt-in, free pin/remove,
  occupied rejection, exclusions/undo, save/close, and window size.
- `python -m compileall capture_dataset` passed.

No NVIDIA GPU, gsplat installation, original Gaussian scene, or actual user YAML/PGM
was available here for a real end-to-end render. The CUDA path calls the same
existing avatar scene loader and camera function, and uses the same gsplat 1.5.3
RGB+ED/packed rendering API. It must be verified on the user's laptop with the pilot.

No claim of semantic accuracy, 3-D surface accuracy, camera clearance, full scene
coverage, real-world metric calibration, or navigation safety follows from these tests.
