# Valmet Field Copilot — final-demo local dashboard

A new `service_demo/` module for your existing `~/SpaceBots_2026/valmet-metaverse` project.
It combines the working avatar navigation, original Gaussian camera, completed DINOv3
catalog, extracted 3D regions, and the attached synthetic Valmet workbook.

**No ROS, RViz, Gazebo, npm, API key, cloud upload, new model, or training required.**
Keep the existing `avatar_demo/`, `feature_dataset/`, `associate_dataset/`,
`segment_dataset/`, `semantic_scene/`, and `semantic_scene_plus/` folders available.
The code imports their existing interfaces without changing them.

## Important workbook finding

The attached workbook is NOT already spatially registered to the boiler-room PLY.
- `START_HERE!B6`: the real Valmet scene had not been received; Scene_Anchors is replaceable.
- `Scene_Anchors!L2:L31`: `placeholder_until_valmet_scene_received`.
- `Scene_Anchors!O2:O31`: remap when the Valmet splat arrives.
- `Knowledge_Docs`, `DOC-3D-MAPPING`: never silently guess asset identity.

The original anchor coordinates, source rows, synthetic flags, state, and records are
preserved. `--demo-bindings` explicitly links workbook asset IDs to spatially spread
extracted regions for this demonstration. These are NOT verified equipment identities.
They are clearly labelled `procedural_demo_binding` in the UI. You can correct them by
selecting a region or placing an explicit demo pin. Original anchors are never overwritten.

A JSON snapshot of the EXACT attached workbook is bundled for the shortest setup path.
It contains 30 master assets, 30 anchors/states, 4,320 telemetry rows, 1,000 events,
300 work orders, 50 historical cases, 8 actual knowledge texts, 38 RAG index entries,
and 120 data-quality records. Sheet/row references and the workbook SHA256 are saved.
The full 20,160-record CSV mentioned in START_HERE and the external markdown documents
listed by RAG_Corpus_Index were not supplied; the app does not pretend to read them.
The sample dates are in October 2026. This is not a live equipment telemetry feed.

## 1. Install (one terminal)

Save `valmet_field_copilot_demo.zip` in Downloads. Stop older GPU-heavy viewers first.

```bash
cd ~/SpaceBots_2026/valmet-metaverse
unzip -n ~/Downloads/valmet_field_copilot_demo.zip -d .
source .venv/bin/activate
```

No dependency installation is required. The app uses the existing NumPy, SciPy,
Pillow, PyYAML, plyfile, PyTorch/gsplat and DINO code. The read-only Excel importer
uses Python's standard ZIP/XML libraries; it does not need openpyxl or Excel.
The launcher does not modify CUDA, drivers, `.bashrc`, or `LD_LIBRARY_PATH`.

## 2. Prepare the dashboard bundle (CPU only)

```bash
bash service_demo/run.sh prepare \
  --out outputs/service_demo/main_v1 \
  --map outputs/ros/valmet_map.yaml \
  --ply "data/Industrial Boiler Room.ply" \
  --regions outputs/semantic_scene/valmet_regions150_v1 \
  --enhancement outputs/semantic_enhance/valmet_regions150_v1 \
  --association outputs/association/valmet_soft200_geometry_v1 \
  --repo vendor/dinov3 \
  --weights models/dinov3_vitb16_pretrain_lvd1689m.pth \
  --demo-bindings
```

This uses the bundled values from your attached workbook. To read a different or
updated original file, add:

```bash
--workbook "data/YOUR_WORKBOOK.xlsx"
```

`--demo-bindings` is an explicit opt-in. Without it assets remain unmapped until you
bind them in the UI. Each automatic binding uses a distinct, non-rejected region
with an approach reachable from the default starting component. If fewer than 30
regions satisfy that test, remaining assets stay unmapped. No paths are invented.
The procedural matching does not assert that a region's shape equals the asset class.

Existing `asset_bindings.json` is never overwritten by a repeated prepare. For changed
source paths/workbook/scene use a new output folder. Mapped asset positions can be inside
occupied cells; the walking endpoint is a separate reachable nearby location.

Check:

```bash
bash service_demo/run.sh check --out outputs/service_demo/main_v1
```

## 3. First launch — navigation and workbook only

```bash
bash service_demo/run.sh serve \
  --out outputs/service_demo/main_v1 \
  --no-render --no-search \
  --port 8770 --open-browser
```

Open `http://127.0.0.1:8770` if needed. Keep the terminal running.
This mode tests the assets, map, navigation, history, controls and source evidence
without loading CUDA or the large DINO catalog. Camera panes intentionally remain blank.
Select an asset, Plan route, then Start. Confirm the map and source crops are correct.
Stop with Ctrl+C before the next launch.

## 4. Full launch — camera + uploaded-image retrieval

```bash
cd ~/SpaceBots_2026/valmet-metaverse
bash service_demo/run.sh serve \
  --out outputs/service_demo/main_v1 \
  --width 480 --height 360 --render-hz 2 \
  --port 8770 --open-browser
```

Only ONE terminal/server is needed. No map_server or point-cloud publisher is needed.
A single spawned worker serializes CUDA tasks: live Gaussian camera, on-demand target
preview, and DINO query encoding. The scene is loaded once. DINO loads lazily on the first
image query. Searching can briefly pause camera updates; the displayed image's position
and age remain visible. Map motion/controls stay in the separate server process.

Keep existing feature weights/vendor source unchanged. If your native checkpoint name
is different, override it when serving:

```bash
--weights "models/YOUR_EXISTING_DINOV3_CHECKPOINT.pth"
```

A unique `models/dinov3_vitb16*.pth` can be found automatically if the default name is
missing. Checkpoint, backbone, dimension and source hashes must match the completed
Stage 3 run. No silent model download or random-weight fallback occurs.

For lower GPU load use `--width 320 --height 240 --render-hz 1`. If only image-query GPU
memory is a problem, try `--search-device cpu` (slower, fp32 query inference may change
scores slightly). Do not change CUDA/PyTorch installations for this demo.

## Layout

- Left ~65%: floor map, avatar pose, target, approach endpoint, path and nearby assets.
- Lower-left tabs: live RGB/depth; target approach preview and source evidence;
  interactive extracted-region point cloud and masked/full-image evidence.
- Right: Assets / Image search / Manual tabs.
- Bottom-right, always visible: speed, camera height, head pitch, turn rate, nearby cards.

The map reads the YAML/PGM directly. Green=input free, gray=unknown, dark=occupied.
Unknown remains blocked for automatic navigation. Previously approved gray *capture*
positions do not become navigable simply because they were used for image generation.
Scroll the map to zoom, drag to pan, click a marker to select it. Use the map-tool menu
to set the start, a manual target, or a selected asset's demo pin.

## Asset mode

1. Select an asset from the dropdown/list or click its map marker.
2. Read state, health, criticality, recommended next check, supplied model confidence,
   three-day trend, recent events, service history, matched historical case and quality flags.
3. Click Plan route. Inspect the path and approach endpoint.
4. Optionally Preview arrival: render the full scene from the planned approach without
   moving the avatar. This image is labelled as a preview, not a live camera frame.
5. Click Start. The avatar turns/walks/stops and faces the target.
6. Nearby demo assets within 2 m are highlighted on the map and in bottom-right cards.
7. Export visit brief saves a JSON evidence package. No work order or equipment command is sent.

**Evidence-led service brief:** a deterministic assembly of workbook facts with exact
sheet/row references and source IDs, NOT a new LLM diagnosis. Historical analogues are
labelled as such. Safety content quotes DOC-FIELD-SAFETY. The app does not derive fault
severity from unprovided thresholds or override the workbook's recorded condition.

Expand **Demo mapping** to choose a different extracted region, or click Place demo pin
and then a map point. This writes only this dashboard's asset_bindings.json. It does not
edit the company workbook, the semantic-scene review file, or original region geometry.

## Image search mode

1. Upload JPG/PNG/WebP of a component. Transparent PNG alpha is used as the mask when available.
2. Optionally drag a crop around the component. Whole-image backgrounds can reduce match quality.
3. Search image. Query encoding uses Stage 3's masked-crop/pooling recipe, crop size and padding.
4. Review five returned original crops and their map locations. IDs include frame + mask + station.
5. Click a result to select it. Inspect its full image/mask overlay; then Plan route and Start.
6. None of these / next five shows the next-ranked candidates; Previous five returns.

The index is the existing Stage 4A catalog: only keep-frame, keep-mask, valid DINO crop,
geometry-eligible observations. It is NOT regenerated or mixed with review masks.
No full N-by-N similarity matrix is created; normalized cosine search runs in CPU chunks.

The minimum cosine defaults to 0.70 and is adjustable per query. This is an uncalibrated
retrieval threshold, NOT probability of identity or an accuracy claim. Below the threshold,
show "No reliable location found" and ask for another image/crop. This is not guaranteed
open-set detection: unrelated images can still produce false candidates. Always review.
At most 200 ranked candidates are retained; each page contains five. At most two masks
from one frame enter the result list, avoiding domination by nested masks in one image.
Multiple views of the same location are allowed. The count of distinct camera stations
within 0.35 m among similar candidates is shown as supporting consistency, not confidence.
No cross-view association or region fusion is rerun by a query.

A result's location is its partial visible-surface median from Stage 2/4A. If that exact
observation belongs to an extracted region, its cloud is available; otherwise no fabricated
object cloud is shown. Appearance matching never silently assigns a workbook asset identity.

The query image, crop/preprocessing, ranked candidate IDs/locations/scores and threshold
are saved in the session. Inputs stay local; no uploads to third parties.

## Manual mode and fixed controls

- Hold Forward/Backward: move along heading, not strafe.
- Hold Left/Right: rotate.
- Keyboard arrow keys perform the same controls in Manual mode.
- W/S: look up/down; H: level. Look controls also work during an automatic walk.
- Escape/Space: stop. Pause button stops automatic/manual motion.
- Input boxes ignore navigation keys while typing.
- Press-and-hold commands require a heartbeat; translation stops within ~0.45 s if input,
  focus or network is lost. Mode changes pause movement.
- Manual movement cancels the old planned path. Plan again from the new location.
- Checked "Restrict manual movement": input-free point-center checking.
- Unchecked: explicit FREE-ROAM / NOCLIP red warning. May cross walls/unknown/exterior.
  This mode is inspection only; it does not edit the map or validate a route.
- Reset/Return to valid start is an explicit teleport, not a demonstrated path.

Speed defaults to 0.25 m/s; turn rate to 60 deg/s; camera height to 1.7 m. The shared
controls can adjust them. Positive pitch looks up, negative looks down. Height changes
camera elevation; it does not implement crouching or a human-body collision model.

## Navigation and rendering conventions

Reuse the working avatar A* planner and stop-turn-walk model. The default is radius=0,
margin=0 to avoid adding a second footprint erosion to your already-inflated map. This
is a DEMO point-center planner, not measured human clearance. Set `--radius 0.25 --margin
0.03` on serve for the previous stricter check. Unknown and occupied cells stay blocked;
diagonal corner-cutting remains prohibited. A target can be occupied; a stopping point
within 0.8–3.5 m is chosen in the current reachable component. Final yaw faces the target.
A path can fail; the UI reports that and does not teleport. Geometric line-of-sight and
full-height/head clearance are NOT verified. Inspect target and route renders.

Original transform retained:
```
x_map = 1.43 * x_GS
y_map = 1.43 * z_GS
z_map = 1.43 * (1.1 - y_GS)
```
Scale/floor are read from the semantic scene manifest; map and geometry must remain
consistent. The usual 1.43 scale is provisional, not a measured physical dimension.
The forward camera uses original gsplat RGB+ED, packed=True, backgrounds=None. Expected
depth is camera-Z, scaled to provisional metres; not a real sensor measurement. The
actual rendered pose is shown by a cyan ring and image-age text. Queueing keeps the
latest live frame instead of claiming an older image is the avatar's current view.

Object cloud pane is a small orbitable point preview (at most 7,000 display points),
not a new point-cloud extraction. It reads cleaned map-frame NPZs when available.
The optional "Saved Gaussian preview" uses existing Stage 4C pre-renders; if absent,
the UI reports that and returns to the source crop. No new on-demand object renderer
is loaded alongside the whole-room camera. Existing Demo 1 remains separate.

## Work products and preservation

```
outputs/service_demo/main_v1/
  config.json
  workbook.json                 # source records + sheet/row references
  asset_bindings.json            # explicit procedural/user demo mappings
  prepare_summary.json
  sessions/<timestamp>/
    config.json
    trajectory.csv
    events.jsonl
    plan_<id>.json
    queries/<id>/query.png
    queries/<id>/results.json
    snapshots/<timestamp>_rgb.png
    snapshots/<timestamp>_depth.npy
    snapshots/<timestamp>_depth_m.npy
    snapshots/<timestamp>_alpha.npy
    snapshots/<timestamp>_camera.json
```

Save camera records the last displayed/rendered pose, not a relabelled current pose.
Export visit brief downloads through the browser. Demo 1 videos, all previous scripts,
PLYs, calibration, masks, descriptors, associations and region archives are untouched.
Do not run upstream reselection or rebuild sources while demonstrating this app.

## Troubleshooting

- **Port in use:** stop the previous server or use `--port 8771`.
- **No assets plotted:** prepare with explicit `--demo-bindings`, or map assets manually.
  Existing bindings are preserved; a new output folder is required for fresh auto bindings.
- **Path fails:** choose a start/approach in the same connected known-free component.
  The source map's unknown corridor is still unknown. Do not silently convert it to free.
- **CUDA unavailable:** first `nvidia-smi`; use the earlier GPU test. The UI remains map-only.
- **Search loading:** catalog checksum verification/loading runs in background; see the badge/terminal.
- **Missing DINO weights:** pass your existing `--weights` and `--repo`; no re-extraction needed.
- **Checkpoint/source mismatch:** use the exact Stage 3 checkpoint/vendor version. Do not bypass hashes.
- **No reliable image match:** crop tightly, use a clearer reference; threshold is adjustable but lower
  values mean more false suggestions. Do not interpret a top score as proof.
- **Camera lag during first query:** lazy model loading/query forward serializes with rendering.
  Query after the initial warm-up for the recorded demo; image age remains visible.
- **High memory:** close old avatar, semantic live-splat viewer, RViz and render jobs.
  Start at 480x360 / 2 Hz. Use `--no-search` or `--no-render` to isolate issues.
- **Missing object appearance:** original source paths/regions must still exist. Use the original
  source camera evidence or cloud; no substitute geometry is invented.

## Focused validation

14 CPU tests passed for workbook counts/anchors/briefs/dates, approach goals, unknown
blocking, noclip distinction, final facing direction, camera inverse transform, retrieval
threshold behavior and atomic writes. The read-only XLSX parser was checked against the
artifact-tool extraction of the actual attached workbook for key sheets including all
telemetry rows; values agreed. Browser UI interaction and map movement were exercised
against the real local server using an artificial spatial fixture and real workbook data.
The sandbox browser required an HTTP-request bridge; this is not GPU validation.
Python compilation and JavaScript syntax passed. Live CUDA, uploaded-query DINO inference
and real-map route correctness must be checked on your laptop.

Run CPU tests (existing upstream modules required):
```bash
python -m unittest discover -s service_demo/tests -v
```

## Recording narrative

1. Assets: choose an attention-needed synthetic asset; show the issue and cited next check.
2. Plan and start: map path, moving avatar, simulated RGB/depth, nearby status cards.
3. Arrival: source-mask evidence, target cloud, and the workbook-grounded service brief.
4. Image search: upload a component crop; review candidates; select and plan.
5. Manual: rotate/look down to demonstrate inspection control.

Use your screen recorder for this dashboard and retain Demo 1 as the separate semantic
extraction/animation clip. This app does not itself encode a browser walkthrough video.

## Source references

- Supplied builder workbook: START_HERE, Asset_Master, Scene_Anchors, Current_State,
  Telemetry_Sample_3d, Alarms_Events, Maintenance_History, Historical_Service_Cases,
  Knowledge_Docs, Data_Quality; workbook hash retained in source records.
- Official gsplat rasterization: https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html
- Official DINOv3: https://github.com/facebookresearch/dinov3
- Local HTTP server scope: https://docs.python.org/3.10/library/http.server.html

Local single-user demonstration only. The standard-library HTTP server is not a
production industrial platform. It binds to 127.0.0.1, validates Host/Origin and a
session token, and serves only whitelisted paths. Do not expose the port publicly.
