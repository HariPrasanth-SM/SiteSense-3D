from pathlib import Path

import numpy as np
from plyfile import PlyData
import open3d as o3d

INPUT = "data/Industrial Boiler Room.ply"
# OUTPUT = Path("outputs/ros/valmet_colored_cloud.ply")
# OUTPUT_NPZ = "outputs/ros/valmet_colored_cloud.npz"

OUTPUT = Path("outputs/ros/valmet_colored_cloud_full.ply")
OUTPUT_NPZ = "outputs/ros/valmet_colored_cloud_full.npz"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Provisional transformation
SCALE = 1.43
FLOOR_Y = 1.1

# MAX_POINTS = 200_000
# MIN_OPACITY = 0.15

MAX_POINTS = 2_000_000
MIN_OPACITY = 0.05

print("Loading Gaussian scene...")

ply = PlyData.read(INPUT, mmap=True)
v = ply["vertex"]

xyz = np.stack([
    np.asarray(v["x"], dtype=np.float32),
    np.asarray(v["y"], dtype=np.float32),
    np.asarray(v["z"], dtype=np.float32)
], axis=1)

dc = np.stack([
    np.asarray(v[f"f_dc_{i}"], dtype=np.float32)
    for i in range(3)
], axis=1)

opacity_logits = np.asarray(
    v["opacity"], dtype=np.float32
)

opacity = 1.0 / (
    1.0 + np.exp(-np.clip(opacity_logits, -30, 30))
)

valid = (
    np.isfinite(xyz).all(axis=1)
    & np.isfinite(dc).all(axis=1)
    & (opacity > MIN_OPACITY)
)

xyz = xyz[valid]
dc = dc[valid]

print("Valid Gaussians:", len(xyz))

# Reproducible sampling
rng = np.random.default_rng(42)

if len(xyz) > MAX_POINTS:
    idx = rng.choice(
        len(xyz), MAX_POINTS, replace=False
    )
    xyz = xyz[idx]
    dc = dc[idx]

# Standard 3DGS DC color approximation
C0 = 0.28209479177387814

rgb = np.clip(
    0.5 + C0 * dc,
    0.0,
    1.0
)

# Gaussian scene -> ROS map coordinates
ros_xyz = np.column_stack([
    SCALE * xyz[:, 0],
    SCALE * xyz[:, 2],
    SCALE * (FLOOR_Y - xyz[:, 1])
])

cloud = o3d.geometry.PointCloud()

cloud.points = o3d.utility.Vector3dVector(
    ros_xyz.astype(np.float64)
)

cloud.colors = o3d.utility.Vector3dVector(
    rgb.astype(np.float64)
)

o3d.io.write_point_cloud(str(OUTPUT), cloud)

print("Saved:", OUTPUT)
print("Points:", len(cloud.points))
print("Bounds:", ros_xyz.min(axis=0), ros_xyz.max(axis=0))

np.savez_compressed(
    OUTPUT_NPZ,
    xyz=ros_xyz.astype(np.float32),
    rgb=(rgb * 255).astype(np.uint8)
)

