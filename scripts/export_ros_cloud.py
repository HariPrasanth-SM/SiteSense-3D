from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

INPUT = "data/Industrial Boiler Room.ply"
OUTPUT = Path("outputs/ros/valmet_cloud.ply")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SCALE = 1.43
FLOOR_Y = 1.1
MAX_POINTS = 150_000

ply = PlyData.read(INPUT, mmap=True)
v = ply["vertex"]

xyz = np.stack([
    np.asarray(v["x"], dtype=np.float32),
    np.asarray(v["y"], dtype=np.float32),
    np.asarray(v["z"], dtype=np.float32),
], axis=1)

opacity = 1 / (
    1 + np.exp(
        -np.clip(
            np.asarray(v["opacity"], dtype=np.float32),
            -30, 30
        )
    )
)

valid = np.isfinite(xyz).all(axis=1) & (opacity > 0.15)

xyz = xyz[valid]

# Reproducible random sample
rng = np.random.default_rng(42)

if len(xyz) > MAX_POINTS:
    indices = rng.choice(
        len(xyz),
        MAX_POINTS,
        replace=False
    )
    xyz = xyz[indices]

# Gaussian coordinates -> ROS coordinates
ros_xyz = np.column_stack([
    SCALE * xyz[:, 0],
    SCALE * xyz[:, 2],
    SCALE * (FLOOR_Y - xyz[:, 1]),
])

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(ros_xyz)

# Temporary neutral color.
# We can later derive RGB from the Gaussian SH DC coefficients.
cloud.paint_uniform_color([0.7, 0.8, 0.9])

o3d.io.write_point_cloud(str(OUTPUT), cloud)

print("Saved:", OUTPUT)
print("Points:", len(ros_xyz))
print("ROS bounds:", ros_xyz.min(axis=0), ros_xyz.max(axis=0))

