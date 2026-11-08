import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.ndimage import binary_dilation, binary_opening

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PLY_PATH = "data/Industrial Boiler Room.ply"
OUT = Path("outputs/maps")
OUT.mkdir(parents=True, exist_ok=True)

RESOLUTION = 0.05  # Scene units per grid cell

X_MIN, X_MAX = -5.5, 5.0
Z_MIN, Z_MAX = -4.5, 4.0

# Initial height bands (positive Y is downward)
FLOOR_Y_MIN = 0.90
FLOOR_Y_MAX = 1.30

OBSTACLE_Y_MIN = -0.20
OBSTACLE_Y_MAX = 0.90

OPACITY_THRESHOLD = 0.15

# Experimental thresholds, not physical occupancy probabilities
FLOOR_THRESHOLD = 0.15
OBSTACLE_THRESHOLD = 0.35

# Inflation radius in grid cells
INFLATION_CELLS = 3

# --------------------------------------------------
# Load Gaussian centers
# --------------------------------------------------

print("Loading Gaussian scene...")

ply = PlyData.read(PLY_PATH, mmap=True)
v = ply["vertex"]

x = np.asarray(v["x"], dtype=np.float32)
y = np.asarray(v["y"], dtype=np.float32)
z = np.asarray(v["z"], dtype=np.float32)

opacity_logits = np.asarray(v["opacity"], dtype=np.float32)
opacity = 1.0 / (
    1.0 + np.exp(-np.clip(opacity_logits, -30, 30))
)

valid = (
    np.isfinite(x)
    & np.isfinite(y)
    & np.isfinite(z)
    & (opacity > OPACITY_THRESHOLD)
)

x = x[valid]
y = y[valid]
z = z[valid]
opacity = opacity[valid]

print(f"Retained {len(x):,} Gaussian centers")

# --------------------------------------------------
# Grid definition
# --------------------------------------------------

x_edges = np.arange(
    X_MIN, X_MAX + RESOLUTION, RESOLUTION
)

z_edges = np.arange(
    Z_MIN, Z_MAX + RESOLUTION, RESOLUTION
)

# Array convention: grid[row_z, col_x]
def make_density(height_min, height_max):
    selected = (
        (y >= height_min)
        & (y < height_max)
    )

    hist, _, _ = np.histogram2d(
        z[selected],
        x[selected],
        bins=[z_edges, x_edges],
        weights=opacity[selected]
    )

    return hist.astype(np.float32)


# --------------------------------------------------
# Floor and obstacle evidence
# --------------------------------------------------

floor_density = make_density(
    FLOOR_Y_MIN,
    FLOOR_Y_MAX
)

obstacle_density = make_density(
    OBSTACLE_Y_MIN,
    OBSTACLE_Y_MAX
)

floor_support = floor_density > FLOOR_THRESHOLD

obstacle_support = (
    obstacle_density > OBSTACLE_THRESHOLD
)

# Remove isolated floor evidence
floor_support = binary_opening(
    floor_support,
    iterations=1
)

# Inflate obstacles conservatively
obstacle_inflated = binary_dilation(
    obstacle_support,
    iterations=INFLATION_CELLS
)

# --------------------------------------------------
# Candidate occupancy
# --------------------------------------------------

# ROS-like convention:
# -1 = unknown
#  0 = candidate free
# 100 = candidate occupied

occupancy = np.full(
    floor_support.shape,
    -1,
    dtype=np.int8
)

occupancy[obstacle_inflated] = 100

candidate_free = (
    floor_support
    & ~obstacle_inflated
)

occupancy[candidate_free] = 0

print("\nMap statistics:")
print("Unknown:", np.sum(occupancy == -1))
print("Free:", np.sum(occupancy == 0))
print("Occupied:", np.sum(occupancy == 100))

# --------------------------------------------------
# Save data
# --------------------------------------------------

np.save(
    OUT / "candidate_occupancy.npy",
    occupancy
)

np.save(
    OUT / "floor_density.npy",
    floor_density
)

np.save(
    OUT / "obstacle_density.npy",
    obstacle_density
)

metadata = {
    "resolution_scene_units": RESOLUTION,
    "origin_x": X_MIN,
    "origin_z": Z_MIN,
    "width": int(occupancy.shape[1]),
    "height": int(occupancy.shape[0]),
    "horizontal_axes": ["x", "z"],
    "vertical_axis": "y",
    "world_up": [0, -1, 0],
    "floor_y_range": [
        FLOOR_Y_MIN,
        FLOOR_Y_MAX
    ],
    "obstacle_y_range": [
        OBSTACLE_Y_MIN,
        OBSTACLE_Y_MAX
    ],
    "occupancy_values": {
        "unknown": -1,
        "free": 0,
        "occupied": 100
    },
    "validated_for_navigation": False
}

with open(
    OUT / "map_metadata.json", "w"
) as f:
    json.dump(metadata, f, indent=2)

# --------------------------------------------------
# Visualizations
# --------------------------------------------------

extent = [
    X_MIN, X_MAX,
    Z_MIN, Z_MAX
]

def save_heatmap(data, filename, title):
    plt.figure(figsize=(11, 9))

    plt.imshow(
        np.log1p(data),
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="equal"
    )

    plt.colorbar(label="Log weighted density")
    plt.xlabel("X (scene units)")
    plt.ylabel("Z (scene units)")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180
    )

    plt.close()


save_heatmap(
    floor_density,
    "floor_support.png",
    "Candidate Floor-Support Evidence"
)

save_heatmap(
    obstacle_density,
    "obstacle_density.png",
    "Candidate Obstacle Evidence"
)

# Occupancy visualization
from matplotlib.colors import ListedColormap

# Convert occupancy to display classes:
# unknown=0, free=1, occupied=2
display = np.zeros_like(occupancy, dtype=np.uint8)
display[occupancy == 0] = 1
display[occupancy == 100] = 2

cmap = ListedColormap([
    "#6b7280",  # unknown
    "#22c55e",  # free
    "#ef4444"   # occupied
])

plt.figure(figsize=(11, 9))

plt.imshow(
    display,
    origin="lower",
    extent=extent,
    cmap=cmap,
    vmin=0,
    vmax=2,
    interpolation="nearest",
    aspect="equal"
)

plt.xlabel("X (scene units)")
plt.ylabel("Z (scene units)")
plt.title(
    "Candidate Occupancy Map\n"
    "Gray=Unknown, Green=Free, Red=Occupied"
)

plt.tight_layout()
plt.savefig(
    OUT / "candidate_occupancy.png",
    dpi=180
)

plt.close()

print("\nSaved outputs to:", OUT)

