import numpy as np
from plyfile import PlyData

path = "data/Industrial Boiler Room.ply"

ply = PlyData.read(path, mmap=True)
v = ply["vertex"]

def stats(name):
    x = np.asarray(v[name], dtype=np.float32)
    x = x[np.isfinite(x)]

    print(
        f"{name:10s} "
        f"min={x.min():.4f} "
        f"p01={np.percentile(x, 1):.4f} "
        f"median={np.median(x):.4f} "
        f"p99={np.percentile(x, 99):.4f} "
        f"max={x.max():.4f}"
    )

print("Gaussian parameters:")
for name in [
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "opacity",
    "rot_0", "rot_1", "rot_2", "rot_3"
]:
    stats(name)

opacity = np.asarray(v["opacity"], dtype=np.float32)
alpha = 1 / (1 + np.exp(-np.clip(opacity, -30, 30)))

print("\nActivated opacity:")
print("Median:", np.median(alpha))
print("Fraction > 0.5:", np.mean(alpha > 0.5))

scales = np.stack([
    np.asarray(v[f"scale_{i}"], dtype=np.float32)
    for i in range(3)
], axis=1)

scales = np.exp(np.clip(scales, -30, 30))

print("\nActivated Gaussian scales:")
print("Median:", np.median(scales, axis=0))
print("99th percentile:", np.percentile(scales, 99, axis=0))

