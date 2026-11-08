import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--view",
        default="outputs/renders/view_A"
    )

    parser.add_argument(
        "--mask",
        required=True
    )

    parser.add_argument(
        "--out",
        default="outputs/pointclouds/object_001.ply"
    )

    parser.add_argument(
        "--min-alpha",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    prefix = args.view

    depth = np.load(prefix + "_depth.npy")
    alpha = np.load(prefix + "_alpha.npy")

    rgb = cv2.imread(prefix + "_rgb.png")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(args.mask)

    if mask.shape != depth.shape:
        mask = cv2.resize(
            mask,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    with open(prefix + "_camera.json") as f:
        camera = json.load(f)

    K = np.array(camera["K"], dtype=np.float64)

    c2w = np.array(
        camera["camera_to_world"],
        dtype=np.float64
    )

    valid = (
        (mask > 127)
        & np.isfinite(depth)
        & (depth > 0)
        & (alpha > args.min_alpha)
    )

    v, u = np.where(valid)

    if len(u) == 0:
        raise RuntimeError("No valid pixels inside mask")

    z = depth[v, u]

    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]

    points_cam = np.stack([x, y, z], axis=1)

    points_world = (
        points_cam @ c2w[:3, :3].T
        + c2w[:3, 3]
    )

    colors = rgb[v, u].astype(np.float64) / 255.0

    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points_world
    )

    cloud.colors = o3d.utility.Vector3dVector(
        colors
    )

    cloud = cloud.voxel_down_sample(0.02)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(str(output), cloud)

    bbox = cloud.get_axis_aligned_bounding_box()

    print("Saved:", output)
    print("Points:", len(cloud.points))
    print("Bounding box center:", bbox.get_center())
    print("Bounding box extent:", bbox.get_extent())


if __name__ == "__main__":
    main()
