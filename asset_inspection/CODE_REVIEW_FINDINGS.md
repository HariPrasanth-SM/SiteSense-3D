# Static review of the supplied code and workbook (no source edits)

## Confirmed in the attached workbook

`START_HERE!B6` says the real scene had not been received and the mapping must be
replaced. All 30 rows in `Scene_Anchors!L2:L31` say
`placeholder_until_valmet_scene_received`. The workbook's original anchor XYZ
ranges are X [-0.32,30.39], Y [-0.39,15.49], Z [0.47,2.15]. These synthetic anchors
are not established correspondences to the scanned room.

## Confirmed in supplied dashboard code

- `service_demo/prepare.py`: procedural candidates are ranked using spatial spread,
  size and coarse reviewed kind. Sorted workbook asset IDs are paired with these
  selected regions. There is NO matching of workbook asset_class to visual type.
- The stored `asset_bindings.json:assets[ID].map_xyz` is already map-frame XYZ.
  `engine.py:resolve_target` and the browser consume it directly. The dashboard
  does not use `anchor_original` as its navigation coordinate.
- Map-click demo pins in `engine.py:command` use `[x,y,1.0]` and no region ID. This
  fixed target height can affect final pitch. It does not establish object extent.
- The saved target is not automatically recomputed from the cleaned region on each
  launch. Cleanup/rebuilds can make a prior binding stale; the audit measures this.

## Transform review

Established formula:

```
p_map = s R p_GS + t
R = [[1,0,0],[0,0,1],[0,-1,0]]
t = [0,0,s*floor_y]
s = 1.43, floor_y = 1.1 (provisional current settings)
```

R is orthonormal with determinant +1. The supplied `camera_matrices`, enhanced
scene transform, world/grid conversion and browser map projection are consistent
with this convention on review. Camera rotations remain rigid; scale applies to
position/depth, not rotation columns. The display correctly compensates image-row
direction. This establishes internal mathematical consistency, not actual live
registration or asset identity.

## One genuine metadata-key risk found upstream

`capture_dataset/__main__.py` stores the floor as `config['floor_y_scene']`.
`semantic_scene/build.py` reads `config.get('floor_y',1.1)` when producing
`scene.json:floor_y_gaussian`.

With the recorded default 1.1, both values coincide. If capture used a different
floor_y_scene, the exporter would silently fall back to 1.1. The inspector compares
original camera records with the dashboard/enhancement manifests to expose this.
This code is deliberately NOT changed in this delivery; existing local manifests
must be checked before applying any correction or rebuilding outputs.

## Do not infer

A close nearest-neighbor distance to any wall is not proof of asset identity. A
failed arrival highlight is not by itself a transform error. Reprojection into
an original supporting view is an internal consistency check, not an independent
validation against measured ground truth.

The actual user-side main_v1 config/bindings and 3D data were not uploaded in this
turn. The standalone script reads those files locally to complete the diagnosis.
