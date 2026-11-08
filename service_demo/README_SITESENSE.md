# SITESENSE 3D — Lavender Concrete dashboard update

An in-place update of the already-working `service_demo/` dashboard. It uses the same
prepared `outputs/service_demo/main_v1` configuration, workbook snapshot, asset bindings,
rendered data, DINOv3 features, regions and Gaussian PLY. **Do not prepare again.**

## Changes

1. Automatic navigation and map-restricted manual movement permit input gray/unknown
   cells as an explicit demo policy. Dark occupied cells, the map exterior, and diagonal
   corner-cutting remain blocked. Only a separate in-memory planning grid is changed.
   The source PGM/YAML, displayed map colors, and previous navigation tools are untouched.
2. On arrival, heading **and pitch** point toward the selected target's XYZ, using the
   current eye height. No body/eye-height change occurs. `Face target` also aims manually.
3. Fixed desktop viewport: map on the left, three persistent RGB/depth/target-crop panels
   in the middle, and the three input modes on the right. No document-level page scroll.
   Long search lists can scroll within their panel; long evidence opens in a bounded modal.
4. Equipment types first, then asset IDs within the selected type; dropdown retained.
   Selected-type map markers have exactly 2.5x their ordinary radius. Other assets remain.
   The 12 control valves fit in a compact 3-column list; change type returns to categories.
5. Plan route / Start / Pause stay at the top of the right column. Route length, eye height,
   and mission name stay visible. Frame/mask IDs remain in saved records and candidate
   tooltips, not in the mission banner. No 'user-confirmed candidate' line.
6. Default title `SITESENSE 3D`; optional `--title "Your title"` on serve.
   Subtitle: `by SpaceBots · Valmet · Industrial Metaverse`.
7. Shared avatar and camera controls form a window-wide bottom footer.
8. Lavender Concrete theme: #EEEAF1 background, #FFFCFF panels, #4E43A4 primary,
   #C36B49 copper, #29263E text, #615C70 secondary, #D8D1DC borders.
   Font stack: Inter, Manrope, then installed system fonts. No remote font/network dependency
   and no bundled font files. Rounded panels, subtle shadows, restrained transitions.
9. Optional one-shot arrival verification: FastSAM mask proposals plus existing DINOv3
   target references and approximate 3D support. A supported candidate appears bright green
   in RGB and as a focused target crop; weak/ambiguous results are not painted green.

## Install — keep the working setup

Stop the old dashboard with Ctrl+C. Save `sitesense3d_dashboard_update.zip` to Downloads.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
cp -a service_demo "service_demo_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/sitesense3d_dashboard_update.zip -d .
source .venv/bin/activate
```

Use **-o** to replace only the updated code files. The patch does not contain your
workbook data, output folders, model weights, map, or other project modules. It keeps
`run.sh`, the workbook importer/snapshot, configuration-preparation code, and common
helpers unchanged. No pip install, CUDA reinstall, or system update is needed.

**Do not run `prepare` again.** Existing mappings and scene paths remain in `main_v1`.

## Launch with the updated UI/navigation, without the optional arrival model check

```bash
bash service_demo/run.sh serve \
  --out outputs/service_demo/main_v1 \
  --title "SITESENSE 3D" \
  --width 480 --height 360 --render-hz 2 \
  --port 8770 --open-browser
```

Only one terminal/server is required. No ROS, RViz, or Gazebo.
Open `http://127.0.0.1:8770` if needed. If an old page is open, use Ctrl+Shift+R.

`--title` may be omitted: SITESENSE 3D is the default. It can be changed without
re-preparing the dataset. For example `--title "SpaceBots | Site Explorer"`.

For a CPU-only layout/navigation test, append `--no-render --no-search`.
To restore conservative gray-cell blocking, append `--block-unknown`.

Default radius/margin remain zero, as in the previously accepted permissive demo.
Passing nonzero `--radius`/`--margin` adds clearance exclusions again. Existing inflation
in your map is not undone. This is a demo point-centre route, not a human collision proof.

## Enable arrival-frame FastSAM + DINOv3 verification

Use **FastSAM**, not FastSLAM. It uses the same existing checkpoint and DINO encoder
as the completed project. No new models are downloaded.

```bash
bash service_demo/run.sh serve \
  --out outputs/service_demo/main_v1 \
  --title "SITESENSE 3D" \
  --width 480 --height 360 --render-hz 2 \
  --arrival-highlight \
  --fastsam FastSAM-s.pt \
  --port 8770 --open-browser
```

`--fastsam` searches the supplied path, then `models/<filename>` and
`weights/<filename>`. An absent checkpoint produces a readable message; navigation
and rendering remain usable. Pass the full existing checkpoint path when necessary.
DINO checkpoint/source identity must still match your feature archive. Existing
`--weights` and `--repo` overrides remain available.

### Exact arrival behavior

- The avatar finishes the route, turns to target XY, and tilts its head toward target Z.
- A fresh final-pose RGB/depth image is rendered.
- FastSAM proposes masks with `retina_masks=True`; mask/RGB alignment is asserted.
- Masks need at least 16 valid depth samples, minimum area 32 pixels, and must not fill
  more than 85% of the frame.
- A mask must have >=15% of sampled surface points within 0.30 provisional metres of
  the stored target surface before DINO is evaluated.
- DINO uses the existing Stage 3 crop/pooling recipe and stored target descriptors.
  Default minimum cosine is 0.72. Up to four crops are encoded in a batch.
- Close-scoring disjoint masks cause abstention. Otherwise the best compatible candidate
  is shown in bright green; the UI calls it a candidate, not confirmed identity.
- Assets use the explicitly demo-bound region's stored geometry/descriptors. Image-search
  targets use the linked region or the selected observation's own mask/depth/descriptor.
  A plain manually typed map point with no reference is not given an invented identity.
- A region outside the view, hidden behind equipment, weakly matching, or lacking sufficient
  geometry yields **no highlight**. Facing the target does not guarantee visibility.
- Final image metadata and target association are preserved. Changing the target/pose
  invalidates old overlays; late worker results from an old request are ignored.

FastSAM runs on CPU to avoid adding a third GPU model to the original scene and DINO on
an 8 GB laptop. DINO retains your configured GPU/CPU device. These operations are serialized
in the existing worker. Live images can pause during verification; the interface and map
remain responsive. No throughput is promised. Disable the flag if the optional check is
not useful for your final demo; all other UI/navigation improvements remain available.

Threshold overrides (heuristic, not calibrated safety/identity confidence):

```text
--arrival-min-similarity 0.72
--arrival-distance 0.30
```

The small **Verify** button in the target-panel header retries the arrival check after
arrival. **Face target** reorients the view, but does not relocate the avatar.

## New layout and interaction

- Map toolbar remains: select a marker, set start, set a manual target, place a selected
  asset pin, reset. Map zoom/pan still work.
- RGB and depth are always visible; there are no live/target/cloud tabs hiding them.
- The third middle panel displays a **tightly cropped masked object**, not the whole
  supporting image. Choose mask crop, original masked crop, or arrival match.
- The point-cloud widget has been removed from this dashboard UI only. Exported clouds,
  semantic viewers, and Demo 1 remain intact.
- Preview arrival opens a modal containing the planned viewpoint; it never replaces the
  persistent target crop with a full-scene image and never moves the avatar.
- Service type selection enlarges all matching asset markers 2.5x. Choose a full asset ID
  from the compact group list or the retained dropdown. Change type opens the type buttons.
- Evidence, telemetry, history, data quality, and mapping are compact single-line controls
  that open readable dialogs. The original source citations and notes are retained there.
- Nearby assets are available through a compact button. Export visit brief still works.
- Footer controls are common to all modes. Arrow keys drive/turn in Manual mode, W/S tilt,
  H levels the view, Escape/Space stop. Navigation keys do not act while typing or reading
  an open modal. Hold controls release on focus loss; server heartbeat expires as before.

Window geometry is intended for desktop use. Tested viewport sizes: 1440x900, 1366x768,
1280x720. The page itself does not scroll; small screens or long content use bounded
panel/dialog scrolling instead of clipping essential information. Browser F11 can provide
extra space when recording. Inter/Manrope are used only when installed locally.

## Map policy: permissive by explicit request, original visualization unchanged

The PGM has 0 = free, -1 = unknown (gray), and 100 = occupied (dark) after loading.
The displayed copy retains all values. The separate routing copy maps unknown to free
only for this demo, leaving occupied unchanged. A* and checked manual motion use it.
Outside the image bounds stays blocked. Diagonal corner cutting stays prohibited.

Routes record how many original unknown cells they cross, and the UI flags gray passage
use. This permission is **not new evidence of free space**. Exterior unknown cells inside
the map rectangle can also be traversed if connected; select your recording routes carefully.
No 3-D/body/head collision or live obstacle sensing is added. A route can still pass under
or through geometry absent from the approximate floor map.

## Files retained / new outputs

No original data or asset bindings are changed by installation or serving.
Each launch still creates a timestamped session. Arrival checks additionally save:

```text
outputs/service_demo/main_v1/sessions/<timestamp>/arrivals/<check_id>/
  view_rgb.png            # RAW final-pose RGB, not painted
  view_depth.npy          # Scene-unit expected depth
  view_depth_m.npy        # Provisional metres
  view_alpha.npy
  view_camera.json
  verification.json       # Pose, target, candidate scores, gate settings
  overlay.png             # Only if a supported candidate is found
  mask.png                # Only if matched
  target_crop.png         # Only if matched
```

Source-mask crop overlays are generated for display, not written over RGB or raw masks.
The prepared workbook JSON, mappings, upstream hashes, and Demo 1 files are untouched.

## Focused validation and limits

- 29 CPU tests passed: 14 existing checks plus 15 update checks.
- Tests include gray corridor connectivity, source-map preservation, occupied/boundary
  blocking, optional strict mode, checked manual movement, target yaw/pitch, tight crops,
  coordinate backprojection, missing references, weak-match abstention, and ambiguous-mask
  abstention. Inference tests use deterministic mocks, not actual pretrained models.
- JavaScript syntax and Python compilation passed.
- Browser smoke exercised grouping, 2.5x markers, source crop, evidence dialog, plan/start/
  pause, manual stop, and viewport sizing against the real local API with an artificial
  map/regions and the supplied workbook snapshot. Sandbox browser network restrictions
  required an HTTP bridge; no production security behavior was disabled in delivered code.
- The preview screenshot uses artificial layout-test geometry and illustrative saved
  images, NOT a live rendering of the user's app in this environment.
- Actual CUDA, FastSAM/DINO inference, real-map path correctness, target visibility and
  recognition quality have NOT been verified on your laptop. Test one mission before recording.

## Rollback

Close the server. Preserve the new code and restore your recorded backup:

```bash
mv service_demo "service_demo_sitesense_saved_$(date +%Y%m%d_%H%M%S)"
cp -a service_demo_backup_YOUR_TIMESTAMP service_demo
```

Your `outputs/service_demo/main_v1` data and all earlier outputs stay separate.

## API references

- FastSAM inference and original-resolution masks:
  https://docs.ultralytics.com/models/fast-sam/
- Official DINOv3 backbone:
  https://github.com/facebookresearch/dinov3

The arrival correspondence gates and abstention logic are prototype heuristics, not an
accuracy-certified industrial perception model. Synthetic asset identities remain demo mappings.
