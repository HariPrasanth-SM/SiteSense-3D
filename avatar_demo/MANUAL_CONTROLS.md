# Valmet avatar — manual exploration update

This update adds local mouse/keyboard movement to the existing avatar demonstration.
It does **not** install packages, change CUDA settings, edit the source map, or
replace `core.py`, `render_worker.py`, or `run.sh`. The existing working rendering
process, scene transform, map loader, and automatic planner are retained.

## Install over the existing package

Close the current avatar window first. Save `valmet_avatar_manual_controls_update.zip`
in Downloads, then:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
cp -a avatar_demo "avatar_demo_backup_$(date +%Y%m%d_%H%M%S)"
unzip -o ~/Downloads/valmet_avatar_manual_controls_update.zip -d .
```

`-o` replaces the included application file. Do not use `-n` for this update: it
would skip the new `app.py`. Your backup preserves the working version.

The update includes `app.py`, `manual.py`, `manual_ui.py`, the README/manual, and
manual-control tests. It relies on the existing base package. No NVIDIA driver,
CUDA, PyTorch, gsplat, ROS, or additional Python package change is needed.

## Launch

Use your existing rendering virtual environment, through your existing launcher:

```bash
cd ~/SpaceBots_2026/valmet-metaverse
bash avatar_demo/run.sh \
  --map outputs/ros/valmet_map.yaml \
  --scene "data/Industrial Boiler Room.ply" \
  --camera-height 1.7 \
  --speed 0.25 \
  --turn-rate 60 \
  --width 480 --height 360 \
  --render-hz 2
```

Two frames per second is a request cap, not a guaranteed measured rate. The
renderer remains a persistent, separate process. Use `--render-hz 1` if necessary;
try `--render-hz 5` only after the controls and images work smoothly. Movement
updates are independent of rendered-image updates. The cyan marker and frame
status retain the actual rendered pose; the image can lag the avatar.

Map-only check:

```bash
bash avatar_demo/run.sh --no-render
```

The speed defaults are now 0.25 m/s and 60 degrees/s. They affect both manual and
automatic motion. The camera remains 1.7 m above the assumed floor. These are
provisional metric values under the existing scale 1.43, not measured dimensions.

## On-screen and keyboard controls

Hold a movement/look button with the mouse; release it to stop that action.
Dragging outside a held button also releases it. These are virtual joystick
buttons; this version does not connect a USB gamepad.

| Action | On-screen control | Keyboard |
|---|---|---|
| Forward in the current heading | Forward | Up arrow |
| Backward along the current heading | Backward | Down arrow |
| Rotate left, without strafing | Turn left | Left arrow |
| Rotate right, without strafing | Turn right | Right arrow |
| Tilt the camera upward | Look up | W |
| Tilt the camera downward | Look down | S |
| Return the camera to level | Level | H |
| Look toward the ceiling | Ceiling +45 degrees | Button |
| Look toward the floor | Floor -45 degrees | Button |
| Stop movement and head tilt | STOP / Pause | Escape or Space |

Up/Down plus Left/Right can be held together to move along a curve. Reverse
movement does not reverse the viewing direction. Pitch changes do not move the
camera vertically: head tilt and camera height remain separate settings.

Head tilt is limited to -80 to +80 degrees, with a default rate of 25 degrees/s.
Use the Pitch field and Apply camera for an exact value. The new `--look-rate`
argument changes the held-button/key tilt rate.

Click the RGB/depth panel to give the application keyboard focus. Arrow keys and
W/S are ignored while you type in coordinate, note, or other entry fields.
Release keys before switching windows. Losing focus or minimizing stops motion.
After Stop, a held key must be released before it can drive again; if a release
happened outside the window, press/release once to clear the guard.

Space is now **stop only**, not auto-resume. Use Walk / resume for a planned route.

## Collision toggle: two different uses

### Restrict manual movement to free cells — ON (default)

By default, manual motion uses the input map's known-free cells: both **green and
amber**. Occupied cells, unknown cells, and map exterior block translation.

This is a **point-centre grid check**, not a human-body collision guarantee. It
removes the additional *manual* footprint restriction without changing the source
map or removing its pre-existing inflation. It does not infer that a small dark
patch is safe to step over. Movement checks every crossed grid cell; it cannot
skip a wall or cut diagonally through a blocked corner.

### Also require body clearance — ON (optional)

Manual motion uses the same strict footprint mask as automatic A*. Only green
cells are eligible. The circular radius and margin still come from `--radius`
and `--margin` (defaults 0.25 m and 0.03 m, plus the existing conservative grid allowance).

### Restrict manual movement to free cells — OFF

This is **free-roam inspection / noclip**. You can move through occupied cells,
unknown cells, walls, equipment, and outside the mapped rectangle. A red warning
banner is shown, the avatar changes color, and unchecked trail segments are pink.

This lets you inspect possible map mistakes. It does **not** show a valid route,
prove that a passage is physically walkable, or automatically approve map cells.
Rendered images inside equipment or beyond reconstructed coverage may be poor.

Both policies preserve the original map. Automatic A* ALWAYS uses the original
strict footprint map, regardless of the manual collision toggle.

### Turning checks back on

If you switch checks ON while standing in a disallowed location, translation
stops; turning and head tilt remain possible. No automatic teleport occurs.

Use either:
- Checks OFF to manually return to a valid area, then turn checks back ON; or
- **Return to free** for an explicit reset/teleport to the last allowed position.

If the saved position fails the newly selected body-clearance policy, Return to
free uses the original planner's candidate auto-start. The toggle itself is not
silently changed. To resume automatic navigation, use a green start and plan a
fresh route. Turn on body clearance before Return to free when needed.

`--noclip` starts with manual map checks OFF. `--manual-strict` starts with the
body-clearance checkbox enabled. Normal launches keep map checks ON and extra
manual body clearance OFF.

## Automatic and manual control do not fight each other

- First Forward/Backward/Left/Right input cancels and clears the old A* route.
- Releasing the manual input leaves the avatar stationary.
- A new target must be planned before automatic walking can resume.
- W/S head tilt can be used while automatic walking continues.
- Changing collision settings pauses motion; it never changes the automatic
  collision map.
- Start clicks are resets/teleports, not evidence of a walked route. With manual
  checks OFF, a start click may be placed in a non-free cell for inspection.

## Save inspection observations for map refinement

The bookmark row saves **your current avatar location**, not the position of an
object seen in the image or the last map location you hovered over.

1. Stop at a useful inspection location.
2. Choose `looks_free`, `obstacle`, `boundary`, or `uncertain`.
3. Enter a note, for example: "Upper corridor looks open; rendered floor visible."
4. Once the camera view has caught up, press **Save bookmark + view**.

Each session is written under:

```text
outputs/avatar_demo/<timestamp>/
├── settings.json
├── trajectory.csv
├── review_points.jsonl
├── bookmarks/
│   └── bookmark_001/
│       ├── bookmark.json
│       ├── view_rgb.png
│       ├── view_depth.npy
│       ├── view_depth_m.npy
│       ├── view_alpha.npy
│       └── view_camera.json
├── snapshots/                # Existing Save camera frame feature
└── plan_001.json              # If an automatic target was planned
```

Map-only mode saves the bookmark without image files. Camera files represent the
most recent displayed frame. If it lags behind your current position, the
bookmark stores both poses and their difference; it never relabels an earlier
image as a current view. Camera JSON keeps its exact height, heading, pitch,
intrinsics, and original Gaussian-world transform.

`trajectory.csv` now additionally records camera pitch/height, manual collision
settings, input map state, footprint classification, and speed settings.
`review_points.jsonl` plus the relevant bookmark folders are the most useful files
to share for boundary review. These annotations are observations to review, not
automatically generated ground truth. No map labels are modified by bookmarking.

## Coordinate convention retained

```text
x_map = 1.43 * x_GS
y_map = 1.43 * z_GS
z_map = 1.43 * (1.1 - y_GS)
```

Heading 0 degrees is map +X; heading 90 degrees is map +Y. Positive pitch looks up;
negative looks down. A 1.7 m camera corresponds to Gaussian Y about -0.0888.
The floor is still assumed flat at Gaussian Y=1.1; this update does not implement
stairs, leg physics, stepping over objects, head-clearance checking, or a fitted
human-body collision volume.

## Recovery / rollback

Close the application before rollback. Preserve the update folder and restore
from the backup made during installation:

```bash
# Replace BACKUP_DIRECTORY with your actual timestamped backup.
mv avatar_demo "avatar_demo_manual_saved_$(date +%Y%m%d_%H%M%S)"
cp -a BACKUP_DIRECTORY avatar_demo
```

The maps, source PLY, CUDA environment, and session outputs are separate and were
not overwritten by this update.

## Validation

49 tests passed in the development environment: 36 CPU core/manual tests and 13
Tk UI tests using a virtual display. They cover collision/no-collision modes,
unknown cells, corner crossing, reverse/turn motion, pitch limits, focus/input
release, autorepeat, automatic-to-manual handover, bookmarks and camera metadata.
An artificial-map UI preview was inspected at 1440x900 for clipped controls.

The CUDA renderer has not been run on the user's laptop as part of this update.
`render_worker.py`, `run.sh`, and `core.py` are byte-identical to the supplied base
archive and are not shipped in the update. See `VALIDATION_MANUAL.md`.
