# Validation before delivery

## Executed successfully in the assistant's working environment

- 14 CPU unit tests, covering:
  - ROS trinary image loading, thresholds and vertical image flip.
  - Cell/world conversion, including nonzero map origin yaw.
  - Unknown-cell blocking and footprint erosion.
  - Rejection of diagonal corner-cutting and disconnected exact goals.
  - A* detours around artificial obstacles.
  - Reachable approach-point selection for an occupied target.
  - Start rejection rather than silently moving a blocked start.
  - Continuous kinematic movement staying in permitted cells.
  - Stopping at the endpoint and facing the target.
  - Replanning from the middle of a path segment.
  - Camera heading, proper rotation, pitch and 1.7 m height conversion.
  - Automatic start selection and fully blocked-map rejection.
- A desktop GUI smoke test under a virtual X display, using an artificial map and
  synthetic gradient image placeholders: window creation, planning, map/pixel
  conversion, image/depth widgets, camera-file saving, walking and shutdown.
- Python syntax compilation of every module and tests.

## Not executed / not established

- No NVIDIA GPU is available in the test environment. The CUDA renderer has NOT
  been executed here. It uses the documented gsplat 1.5.3 API and the same standard
  PLY conventions used in your previously working render script.
- The actual Valmet PLY and map arrays/YAML are not present here. Navigation and
  camera views have NOT been validated against your industrial environment.
- No FPS, memory budget or full-scene runtime is guaranteed.
- Physical dimensions, head clearance, occlusion-free approach and collision-free
  movement in the actual 3-D scene have not been established by these tests.

Start with `--check-map`, then `--no-render`, then a short rendered route on your
laptop. Preserve your existing scripts and use the new package as an additional
prototype, not a replacement for your working ROS setup.
