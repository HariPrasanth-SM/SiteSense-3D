# Sources and acknowledgements

**Challenge and inputs:** Valmet Automation, Since AI Hackathon 2026, *Industrial Metaverse: Semantic 3D Scene Understanding & Autonomous Field-Service Copilot*. The supplied challenge brief asks for navigable space, semantic asset extraction, metadata integration, and a service walkthrough. Section 02 and the mapping note in Section 05 explicitly permit procedural or arbitrary mapping of synthetic workbook records to scene objects.

**Models and software:**
- [gsplat](https://github.com/nerfstudio-project/gsplat): Gaussian rasterization and simulated RGB/expected-depth views.
- [FastSAM / Ultralytics](https://docs.ultralytics.com/models/fast-sam/): image mask proposals.
- [DINOv3](https://github.com/facebookresearch/dinov3): visual descriptors for image retrieval and cross-view candidate matching.
- NumPy, SciPy, Pillow, PyYAML, plyfile, PyTorch, and browser canvas/WebGL utilities used throughout the prototype.
- Open3D and FFmpeg are used by optional inspection/video tools. OpenCLIP category suggestions are optional and are not the basis of the final 17 manual asset identities.

The diagrams in `docs/diagrams/` are AI-generated conceptual illustrations with separate editable SVG block-diagram counterparts. They do not depict measured output from the supplied scene. No font files or pretrained model weights are bundled.

Company-provided geometry, workbook data, and derived assets must be shared according to the provider's permission. Dependency and model licenses remain separate from any license the team chooses for its own code. This kit does not assign a project-wide license.
