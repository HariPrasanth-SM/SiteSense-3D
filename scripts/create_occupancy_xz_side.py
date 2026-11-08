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
horizontal = xyz[:, [0, 1]]
height = xyz[:, 2]

print("Height percentiles:")
print(np.percentile(height, [1, 5, 25, 50, 75, 95, 99]))

# Top-down density visualization.
resolution = 0.05  # scene units per cell

x_min, y_min = horizontal.min(axis=0)
x_max, y_max = horizontal.max(axis=0)

x_edges = np.arange(x_min, x_max + resolution, resolution)
y_edges = np.arange(y_min, y_max + resolution, resolution)

density, _, _ = np.histogram2d(
    horizontal[:, 0],
    horizontal[:, 1],
    bins=[x_edges, y_edges],
    weights=opacity
)

plt.figure(figsize=(12, 10))

plt.imshow(
    np.log1p(density.T),
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="inferno",
    aspect="equal"
)

plt.colorbar(label="Log weighted Gaussian density")

plt.xlabel("X (scene units)")
plt.ylabel("Y (scene units)")
plt.title("Valmet Boiler Room — Side Gaussian Density")

plt.tight_layout()

plt.savefig(
    OUT / "gaussian_density_XY.png",
    dpi=200
)

print("Saved:", OUT / "gaussian_density_XY.png")
