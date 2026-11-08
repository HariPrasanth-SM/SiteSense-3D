# SITESENSE 3D — manually labelled service objects

Patch for the last completed SITESENSE dashboard (`service_demo/`), not the abandoned
scale/flip experiment. No new scale, origin-offset or flip flags are introduced.

## What changes

- Service missions, persistent asset map markers, groups, dropdown, nearby cards and
  service briefs use ONLY the active objects from the manual annotations.
- Each label joins the synthetic workbook by exact tag (case, dash and leading-zero
  normalization is permitted; no fuzzy equipment-identity or category-only assignment).
- Uses `target_map_xyz` as the aim point and `bbox_map` as the recorded box. It never
  replaces the aim point with a box center or with an Excel placeholder anchor.
- Preserves non-contiguous tags and original `manual_####` IDs.
- Canonical display spelling comes from the exact workbook tag/class; original label
  and category are retained and any change is listed in the import report.
- The supplied file has 17 active objects, all 17 exactly matching workbook tags.
  No spelling correction was necessary in this file. No object is merged or relocated.
- Positions/boxes are human observations from rendered depth; service conditions,
  telemetry and history are still SYNTHETIC workbook records, joined by user-assigned
  tag. The workbook records are not newly measured facts about the photographed object.
- Existing image search and manual exploration remain available. Image search still
  queries the completed DINO catalog; its candidates do not reintroduce the old 30
  placeholder asset markers.
- Automatic walking resets pitch to 0 degrees BEFORE motion and holds it level during
  route following and turning. At arrival the existing yaw/pitch aiming uses the selected
  target's XYZ and current eye height. A new walk resets it again. Eye height is unchanged.
- Pitch edits and look keys are ignored during automatic travel; stop or use Manual mode
  to tilt freely. This avoids carrying the previous target's head tilt into the next walk.
- RGB/Depth/Target headings are thin LEFT rails, rotated 90 degrees counterclockwise.
  There are no top header rows subtracting from image height. Save/target controls and
  image-age text appear as small hover/focus overlays. Click an image to enlarge it.
- Full camera aspect ratio is preserved with `object-fit: contain` (no artificial stretch
  or blind crop). The target is a focused manual object crop, not the full scene.
- Read-only annotation details replace the old procedural remapping controls. Overlapping
  map markers open a small selection list rather than silently selecting the first one.
- Existing gray-cell demo navigation policy is retained; source map files are unchanged.

## Supplied labels

| Category | Tags | Count |
|---|---|---:|
| Control Valve | CV-002, CV-004, CV-005, CV-006, CV-008 | 5 |
| Electric Motor | M-001, M-002, M-003, M-005 | 4 |
| Process Pump | P-002, P-003, P-004 | 3 |
| Pressure Transmitter | PT-001, PT-003 | 2 |
| Vibration Sensor | VS-001, VS-002, VS-003 | 3 |

P-002 and P-004 aim points are about 0.118 m apart in 3-D in the supplied annotation.
Both are preserved. This is a review flag, not an automatic duplicate decision. When
markers overlap, choose the tag in the click-selection list or the service dropdown.

## 1. Install the patch

Stop the running dashboard with Ctrl+C. Save `sitesense_manual_assets_update.zip` in Downloads.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
cp -a service_demo "service_demo_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/sitesense_manual_assets_update.zip -d .
source .venv/bin/activate
```

This replaces only the files in this patch. No pip, system, CUDA or driver change.
Keep your existing `service_demo/`, `avatar_demo/`, other upstream modules and data.
Do not run the old `prepare --demo-bindings` command: that would recreate the workflow
we are replacing.

## 2. Import labels into a NEW prepared bundle

Use the ORIGINAL manual-label output folder. Its `objects/` directory contains images,
clouds and masks referenced by annotations.json. The JSON alone contains their paths,
not their bytes.

```bash
bash service_demo/run.sh import-labels \
  --base outputs/service_demo/main_v1 \
  --annotations outputs/manual_labels/scene_v1/annotations.json \
  --labels-root outputs/manual_labels/scene_v1 \
  --out outputs/service_demo/manual_v1
```

Change the labels folder if you used another name. The importer reads the workbook
snapshot already prepared in `main_v1`; it does not require reimporting the Excel file.
The old bundle and old asset bindings are not overwritten.

If the JSON is elsewhere, point `--annotations` there and still set `--labels-root` to
its ORIGINAL folder containing `objects/`. The exact JSON supplied in chat is also
included as `service_demo/data/manual_annotations_supplied.json`, usable as a fallback
for --annotations. It does not include the local image/cloud files.

The importer:
1. Checks the source scene/map hashes when present, the map-frame convention, and the
   existing scale=1.43 / floor_y=1.1 values; mismatches are errors, not silently corrected.
2. Checks that map and original-scene target positions agree under the existing transform.
3. Matches active annotations to workbook tags and writes only the matched 17 records.
4. Records missing evidence paths as warnings without borrowing unrelated old-region crops.
5. Writes the new bundle atomically, refuses to overwrite an existing output folder, and
   leaves the original annotation JSON, images, workbook and main_v1 unchanged.

Expected summary: 17 labelled objects, 0 spelling corrections for the supplied file.
Open `outputs/service_demo/manual_v1/manual_import_report.json` to inspect all joins.
An occupied aim-point cell is not itself an error: equipment is an obstacle and the
planner finds a separate nearby stopping location.

## 3. Check the new bundle

```bash
bash service_demo/run.sh check --out outputs/service_demo/manual_v1
```

Expected counts:
- assets: 17
- mapped: 17
- manual_objects: 17
- manual_crops_found: 17 when the labeler `objects/` files are present
- travel_pitch_deg: 0.0

`manual_crops_found` lower than 17 means missing local evidence or an incorrect
--labels-root. Navigation can work without a crop, but the target pane shows a clear
missing-file notice instead of an unrelated image. Fix by importing a NEW bundle with
the correct root; do not move the original evidence while serving.

## 4. Launch the final dashboard

```bash
bash service_demo/run.sh serve \
  --out outputs/service_demo/manual_v1 \
  --title "SITESENSE 3D" \
  --width 640 --height 480 --render-hz 2 \
  --port 8770 --open-browser
```

Keep this one terminal open. Browser: `http://127.0.0.1:8770`.
Hard-refresh an existing browser tab using Ctrl+Shift+R.

640x480 gives a sharper source image and enlarged-view dialog, but does not change
camera intrinsics conventions. Use 480x360 if you prefer your original lighter settings:

```bash
bash service_demo/run.sh serve \
  --out outputs/service_demo/manual_v1 \
  --width 480 --height 360 --render-hz 2 \
  --port 8770 --open-browser
```

Quick UI-only check: append `--no-render --no-search`. Normal image search reuses
existing DINO features/models and is not re-extracted. No ROS/Gazebo needed.

## Navigation sequence

1. Select a labelled object (for example PT-001) from its type group or a map marker.
2. Plan route: object aim point stays the annotated XYZ; stopping point is separate.
3. Start: pitch becomes 0 immediately, then the avatar moves and turns along the path.
4. Arrival: yaw and pitch aim toward the annotated object point.
5. Select another object, Plan route, Start: pitch resets to 0 again before moving.

Automatic navigation keeps the existing static 2D collision checks and final aiming;
it does not validate head clearance, hidden geometry or object visibility. The source
annotations themselves mark navigation goals as unverified. No extra coordinate flips,
rescaling or origin offsets are introduced.

## Reference images and optional arrival check

The Target panel reads the reference observation from the manual annotations, crops
it tightly using the saved mask, and never falls back to the previous arbitrarily
assigned extracted region. The cropped RGBA reference is used when source RGB/mask
files are unavailable but the crop exists. User-facing imagery is read-only.

The existing optional `--arrival-highlight --fastsam FastSAM-s.pt` remains available.
For a manual target, its own RGBA reference crop is encoded with the existing DINO
recipe on demand; its own saved map-frame cloud supplies geometric support. It does
not reuse the old demo-bound object's descriptor. FastSAM inference and strict
appearance/geometric gates remain as before, with abstention on missing or weak
references. Normal launch does not run this optional verification. Actual model
inference for this extension still needs checking on the user's laptop.

## Files created in the new bundle

```text
outputs/service_demo/manual_v1/
  config.json                  # Existing scene/map transform; manual catalog enabled
  workbook.json                # Only 17 matching asset records and their service data
  asset_bindings.json          # Exact manually annotated targets; no old region IDs
  manual_annotations.json      # Unmodified copy of your supplied annotation file
  manual_assets.json           # Joined geometry, evidence paths, original/canonical names
  manual_import_report.json    # Matches, spelling changes, warnings and missing media
  manual_asset_matches.csv
  sessions/                    # Existing trajectory, plans, snapshots and query outputs
```

All supporting cameras/clouds stay in the original labels-root. Existing raw workbook
anchors remain as provenance but do not influence map placement, aiming or path goals.

## Focused validation performed

- 13 new focused CPU tests: supplied 17-tag join, exact target/box preservation,
  workbook-state/telemetry preservation, category/tag normalization with audit trails,
  duplicate/unknown-tag refusal, scale/frame mismatch refusal, same-coordinate objects,
  source-path confinement, compact mask crop, atomic separate bundle creation, and
  level-travel / arrival / next-leg reset.
- 29 existing dashboard CPU tests passed unchanged.
- Python compilation and JS syntax checked.
- Browser smoke against the actual local API with artificial map/image fixtures:
  17 visible assets, selection/crop endpoint, plan/start resets -30° to 0°, read-only
  annotation detail, no accidental remapping, counterclockwise rails, and no document
  overflow at 1366x768, 1280x720 and 1440x900.
- Browser network restrictions required a test-only Python HTTP bridge; delivered
  Host/Origin/session-token protections are unchanged.
- The layout screenshot uses SYNTHETIC UI-test geometry/images, not live Valmet output.
- No live CUDA rendering, new FastSAM/DINO inference or physical scene-route validation
  was run here. No claim of measured physical equipment identity or clearance is made.

## Rollback

Stop server, preserve updated code, then restore the backup folder:

```bash
mv service_demo "service_demo_manual_saved_$(date +%Y%m%d_%H%M%S)"
cp -a service_demo_backup_YOUR_TIMESTAMP service_demo
```

Launch your original `--out outputs/service_demo/main_v1` bundle. Its files were never
changed. The newer manual_v1 bundle and original labeler outputs remain preserved.
