import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization


def load_gaussians(path, device="cuda"):
    ply = PlyData.read(path, mmap=True)
    v = ply["vertex"]

    def field(name):
        return np.asarray(v[name], dtype=np.float32).copy()

    means = np.stack(
        [field("x"), field("y"), field("z")], axis=1
    )

    scales = np.exp(
        np.stack([field(f"scale_{i}") for i in range(3)], axis=1)
    )

    quats = np.stack(
        [field(f"rot_{i}") for i in range(4)], axis=1
    )
    quats /= np.maximum(
        np.linalg.norm(quats, axis=1, keepdims=True), 1e-8
    )

    opacities = 1.0 / (
        1.0 + np.exp(-np.clip(field("opacity"), -30, 30))
    )

    # Standard 3DGS stores SH coefficients channel-major.
    dc = np.stack(
        [field(f"f_dc_{i}") for i in range(3)], axis=1
    )[:, None, :]

    rest = np.stack(
        [field(f"f_rest_{i}") for i in range(45)], axis=1
    ).reshape(-1, 3, 15).transpose(0, 2, 1)

    sh = np.concatenate([dc, rest], axis=1)

    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    return (
        tensor(means),
        tensor(quats),
        tensor(scales),
        tensor(opacities),
        tensor(sh),
    )


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    # Camera-to-world with OpenCV camera convention:
    # +X right, +Y down, +Z forward
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye

    return c2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ply",
        default="data/Industrial Boiler Room.ply"
    )
    parser.add_argument(
        "--out",
        default="outputs/renders/view_001"
    )
    parser.add_argument(
        "--width", type=int, default=640
    )
    parser.add_argument(
        "--height", type=int, default=480
    )
    parser.add_argument(
        "--eye", nargs=3, type=float,
        default=[0, 0, -3]
    )
    parser.add_argument(
        "--target", nargs=3, type=float,
        default=[0, 0, 0]
    )
    parser.add_argument(
        "--up", nargs=3, type=float,
        default=[0, 1, 0]
    )
    parser.add_argument(
        "--fov", type=float, default=70
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Gaussian scene...")
    means, quats, scales, opacities, sh = load_gaussians(
        args.ply
    )

    c2w = look_at(args.eye, args.target, args.up)
    w2c = np.linalg.inv(c2w)

    H, W = args.height, args.width

    fy = H / (2 * np.tan(np.deg2rad(args.fov) / 2))
    fx = fy

    K = np.array([
        [fx, 0, W / 2],
        [0, fy, H / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    viewmats = torch.from_numpy(
        w2c[None]
    ).cuda()

    Ks = torch.from_numpy(
        K[None]
    ).cuda()

    print("Rendering...")

    with torch.no_grad():
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            sh_degree=3,
            render_mode="RGB+ED",
            near_plane=0.01,
            far_plane=100.0,
        )

    rgb = renders[0, :, :, :3].cpu().numpy()
    depth = renders[0, :, :, 3].cpu().numpy()
    alpha = alphas[0, :, :, 0].cpu().numpy()

    rgb_u8 = (
        np.clip(rgb, 0, 1) * 255
    ).astype(np.uint8)

    Image.fromarray(rgb_u8).save(
        str(out) + "_rgb.png"
    )

    # Save raw float32 depth for back-projection.
    np.save(
        str(out) + "_depth.npy",
        depth.astype(np.float32)
    )

    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & (alpha > 0.2)
    )

    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [2, 98])
        depth_vis = np.zeros_like(depth)

        if hi > lo:
            depth_vis[valid] = np.clip(
                (depth[valid] - lo) / (hi - lo),
                0, 1
            )

        depth_vis = (
            depth_vis * 255
        ).astype(np.uint8)

        Image.fromarray(depth_vis).save(
            str(out) + "_depth.png"
        )

    np.save(
        str(out) + "_alpha.npy",
        alpha.astype(np.float32)
    )

    with open(str(out) + "_camera.json", "w") as f:
        json.dump({
            "K": K.tolist(),
            "camera_to_world": c2w.tolist(),
            "world_to_camera": w2c.tolist(),
            "width": W,
            "height": H,
            "depth_type": "expected_camera_z",
            "coordinate_convention": "OpenCV"
        }, f, indent=2)

    print("Saved:")
    print(str(out) + "_rgb.png")
    print(str(out) + "_depth.npy")
    print(str(out) + "_depth.png")
    print(str(out) + "_alpha.npy")
    print(str(out) + "_camera.json")


if __name__ == "__main__":
    main()
