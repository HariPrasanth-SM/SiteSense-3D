# Setup

## Existing demo laptop

Use [RUN_DEMO.md](../RUN_DEMO.md). Preserve the working `.venv`, CUDA toolkit, model revision, checkpoints, and prepared data. No installation is needed just to publish the repository.

## Fresh Ubuntu environment

The development machine used Ubuntu 22.04 / Python 3.10, CUDA Toolkit 12.6, an RTX 4060 Laptop GPU (8 GB VRAM), and 16 GB system RAM. Those are development details, not minimum-performance guarantees.

```bash
# From the cloned source repository:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
```

For live RGB/depth, install CUDA-enabled PyTorch/torchvision appropriate to your driver and toolkit, then install the project's renderer version:

```bash
python -m pip install gsplat==1.5.3
```

Choose a PyTorch release that supports Python 3.10 and matches the recorded environment; do not assume the newest release is compatible. Follow the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [gsplat installation guide](https://github.com/nerfstudio-project/gsplat#installation). CUDA extension compilation may require a matching toolkit, a host C++ compiler, and Ninja. `nvidia-smi` reports driver capability; it does not identify the installed `nvcc` toolkit.

The staging helper records package versions and the DINO repository commit under `environment/` when run from the working virtual environment. Use that inventory when reproducing the result. Do not reinstall the author's working GPU stack merely because this guide lists a version.

ROS, Gazebo, npm, hosted APIs, and cloud services are not required. Tkinter is needed only for older desktop tools. FFmpeg is needed only for video encoding. Ultralytics is needed for FastSAM processing/optional arrival checks, not for displaying completed asset missions.

## Restore runtime data

Restore the folders listed in [DATA.md](DATA.md). Source-only Git clones do not contain the room splat, saved image-search cache, or original manual evidence.

For a **different checkout path**, prepare a new asset/manual bundle using the restored inputs:

```bash
bash service_demo/run.sh prepare \
  --out outputs/service_demo/base_local \
  --map outputs/ros/valmet_map.yaml \
  --ply "data/Industrial Boiler Room.ply" \
  --regions outputs/semantic_scene/valmet_regions150_v1 \
  --enhancement outputs/semantic_enhance/valmet_regions150_v1 \
  --association outputs/association/valmet_soft200_geometry_v1

bash service_demo/run.sh import-labels \
  --base outputs/service_demo/base_local \
  --annotations outputs/manual_labels/scene_v1/annotations.json \
  --labels-root outputs/manual_labels/scene_v1 \
  --out outputs/service_demo/manual_local

bash service_demo/run.sh serve \
  --out outputs/service_demo/manual_local \
  --title "SITESENSE 3D" --width 640 --height 480 --render-hz 2 \
  --no-search --port 8770 --open-browser
```

The first command needs either an authorized `service_demo/data/workbook_snapshot.json` or `--workbook "path/to/the/provided/workbook.xlsx"`. The public staging helper omits the bundled company snapshot unless explicitly requested. Do **not** add `--demo-bindings`: the manual import supplies the real annotated positions.

Use new output directory names when these already exist. Keep annotation and map identities consistent; no scale or flip correction is part of this setup.

## Image search

Install the optional retrieval data from [DATA.md](DATA.md), use the exact official DINOv3 code revision recorded by feature extraction, and obtain authorized ViT-B/16 weights through the [official DINOv3 project](https://github.com/facebookresearch/dinov3). No model weights are included in this repository.

The live server accepts `--repo vendor/dinov3` and `--weights models/YOUR_CHECKPOINT.pth`. Matching source/checkpoint hashes are checked. A new clone with a different DINO revision is not automatically equivalent to the feature-extraction environment.

Remove `--no-search` only after its catalog and referenced source paths are available. Optional `--arrival-highlight --fastsam FastSAM-s.pt` additionally requires the local FastSAM checkpoint and Ultralytics.

## Additional tools

Each module exposes `bash MODULE/run.sh --help` and its own README. Typical entry points are `manual_labeler/run.sh serve`, `semantic_scene_plus/run.sh view`, and the `semantic_animation/` / `semantic_showcase/` preparation, rendering, and encoding commands. Their source-data prerequisites are described in their module guides.
