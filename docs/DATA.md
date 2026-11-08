# Source repository and data downloads

## What belongs in GitHub?

| Include | Keep outside Git |
|---|---|
| Python modules, launchers, browser HTML/CSS/JS | `.venv/`, caches, compiled extensions, old backups |
| Short README, run sheet, architecture figures | Original room PLY and large environment clouds |
| Current map YAML/PGM and a small annotation example | Per-object PLY/NPZ/Gaussian bundles |
| A few chosen image-query examples and a short demo GIF | Model checkpoints, full captures, mask/feature caches |
| Lightweight configuration examples and dependency inventory | Sessions, uploads, trajectories, full videos and ZIPs |

Do not publish the workbook, the scene, or derived content beyond the permissions provided for the challenge.

## Download A: inspect the extracted objects

Use `tools/package_object_assets.py`. This contains individual object clouds and Gaussian subsets, metadata, selected evidence, and an optional environment point cloud. It can be inspected without re-running the extraction pipeline.

A point-cloud PLY is not a Gaussian PLY. `surface_clean.ply` is in the map frame; `gaussians_original.ply` retains the original Gaussian axes and attributes. The manifest records the transform. Regions can overlap and are not guaranteed distinct complete physical assets.

**Drive:** `GOOGLE_DRIVE_OBJECT_ASSETS_URL` — replace before publishing.

## Download B: run asset missions and manual exploration

Preserve these paths from the working project in a separate runtime archive:

```text
data/Industrial Boiler Room.ply
outputs/ros/valmet_map.yaml
outputs/ros/valmet_map.pgm
outputs/service_demo/manual_v1/      # omit sessions/
outputs/manual_labels/scene_v1/      # include objects/, not only annotations.json
outputs/semantic_scene/valmet_regions150_v1/
outputs/semantic_enhance/valmet_regions150_v1/
```

The current dashboard still reads the semantic-scene manifest and region metadata at startup, even when image search is disabled. Do not assume the 17-label JSON alone is sufficient. A conservative handoff retains the two region folders; large exported files can be reduced later without changing the working demo now.

Use `--no-search` for this mode. No FastSAM/DINO inference archives are needed for asset/manual navigation alone. The room PLY is needed for RGB/depth; `--no-render` disables that part.

## Optional Download C: keep image-based search working

The current implementation needs more than the object ZIP:

| Source | Runtime records to preserve |
|---|---|
| Association folder | `catalog_complete.json`, `catalog.jsonl`, `crop_vectors.npy`, `frame_bindings.json`, `source.json`, and any other files named in `catalog_complete.json:files` |
| Feature folder | At least the **unchanged** `run.json` and `summary.json` referenced by the catalog |
| Capture folder | **Unchanged** `plan.json` and source-frame `rgb.png`; retain `camera.json`, `depth_m.npy`, and `alpha.npy` for geometric arrival checks |
| Segmentation folder | `quality_policy.json` and source-frame `masks.npz` used to display retrieved crops/mask overlays |
| Local model setup | The matching official DINOv3 source revision and authorized checkpoint |

The full dense DINO patch cache and all per-mask feature archives are **not read by the basic live lookup**: the prepared `crop_vectors.npy` already contains its retrieval descriptors. Keep the larger archives privately for reprocessing, not necessarily in the public download.

However, **saved masks and RGB frames are still read by search-result thumbnails and overlays**. Removing them breaks that feature. FastSAM checkpoint/inference is only additionally needed when running segmentation or optional arrival highlighting.

## Portability

The current prepared files contain absolute laptop paths. Copying them beside a fresh clone does not automatically relocate those paths. For asset/manual mode, create a fresh bundle using the commands in [SETUP.md](SETUP.md). For search, keep the catalog's referenced data at consistent paths or perform a deliberate relocation/rebuild; do not blindly replace strings in hashed plans and source records.

This release documents the working laptop demo. It is not a claim that the source-only repository runs the live scene without the separately supplied data and models.
