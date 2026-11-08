import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from pathlib import Path

PLY_PATH = "data/Industrial Boiler Room.ply"

OUT = Path("outputs/maps")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading Gaussian scene...")

ply = PlyData.read(PLY_PATH, mmap=True)
v = ply["vertex"]

xyz = np.stack([
    np.asarray(v["x"]),
    np.asarray(v["y"]),
    np.asarray(v["z"])
], axis=1)

opacity_logits = np.asarray(v["opacity"])
opacity = 1.0 / (1.0 + np.exp(-opacity_logits))

# Filter very low-opacity Gaussians for visualization.
keep = opacity > 0.15
xyz = xyz[keep]
opacity = opacity[keep]

print("Retained Gaussian centers:", len(xyz))

# Provisional assumption: Y is vertical.
horizontal = xyz[:, [0, 2]]
height = xyz[:, 1]

print("Height percentiles:")
print(np.percentile(height, [1, 5, 25, 50, 75, 95, 99]))

# Top-down density visualization.
resolution = 0.05  # scene units per cell

x_min, z_min = horizontal.min(axis=0)
x_max, z_max = horizontal.max(axis=0)

x_edges = np.arange(x_min, x_max + resolution, resolution)
z_edges = np.arange(z_min, z_max + resolution, resolution)

density, _, _ = np.histogram2d(
    horizontal[:, 0],
    horizontal[:, 1],
    bins=[x_edges, z_edges],
    weights=opacity
)

plt.figure(figsize=(12, 10))

plt.imshow(
    np.log1p(density.T),
    origin="lower",
    extent=[x_min, x_max, z_min, z_max],
    cmap="inferno",
    aspect="equal"
)

plt.colorbar(label="Log weighted Gaussian density")

plt.xlabel("X (scene units)")
plt.ylabel("Z (scene units)")
plt.title("Valmet Boiler Room — Top-Down Gaussian Density")

plt.tight_layout()

plt.savefig(
    OUT / "gaussian_density.png",
    dpi=200
)

print("Saved:", OUT / "gaussian_density.png")
