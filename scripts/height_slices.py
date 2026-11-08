from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

PLY_PATH = "data/Industrial Boiler Room.ply"
OUT = Path("outputs/maps")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading PLY...")

ply = PlyData.read(PLY_PATH, mmap=True)
v = ply["vertex"]

x = np.asarray(v["x"], dtype=np.float32)
y = np.asarray(v["y"], dtype=np.float32)
z = np.asarray(v["z"], dtype=np.float32)

opacity_logits = np.asarray(v["opacity"], dtype=np.float32)
opacity = 1.0 / (
    1.0 + np.exp(-np.clip(opacity_logits, -30, 30))
)

# Filter only very low-opacity Gaussians.
valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(z)
    & (opacity > 0.15)
)

x = x[valid]
y = y[valid]
z = z[valid]
opacity = opacity[valid]

print("Retained Gaussians:", len(x))

# Determine scene height distribution.
print("\nY percentiles:")
print(np.percentile(y, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]))

# Fixed spatial extent for comparison.
x_min, x_max = -5.5, 5.0
z_min, z_max = -4.5, 4.0

resolution = 0.05

x_edges = np.arange(x_min, x_max + resolution, resolution)
z_edges = np.arange(z_min, z_max + resolution, resolution)

# Equal-width slices across the central height distribution.
y_low, y_high = np.percentile(y, [1, 99])
edges = np.linspace(y_low, y_high, 9)

fig, axes = plt.subplots(2, 4, figsize=(18, 10))

for i, ax in enumerate(axes.flat):

    lo, hi = edges[i], edges[i + 1]

    selected = (y >= lo) & (y < hi)

    density, _, _ = np.histogram2d(
        x[selected],
        z[selected],
        bins=[x_edges, z_edges],
        weights=opacity[selected]
    )

    ax.imshow(
        np.log1p(density.T),
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],
        cmap="inferno",
        vmin=0,
        vmax=5,
        aspect="equal"
    )

    ax.set_title(
        f"Slice {i + 1}: Y [{lo:.2f}, {hi:.2f}]"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    print(
        f"Slice {i+1}: Y [{lo:.2f}, {hi:.2f}], "
        f"Gaussians: {selected.sum():,}"
    )

plt.suptitle(
    "Valmet Boiler Room — Height-Filtered Gaussian Density",
    fontsize=16
)

plt.tight_layout()

output = OUT / "height_slices.png"
plt.savefig(output, dpi=180)

print("\nSaved:", output)

