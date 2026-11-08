from pathlib import Path
from plyfile import PlyData
import numpy as np

path = Path("data/Industrial Boiler Room.ply")

if not path.exists():
    raise FileNotFoundError(f"PLY file not found: {path}")

print(f"File: {path}")
print(f"Size: {path.stat().st_size / 1024**3:.2f} GB")

ply = PlyData.read(str(path), mmap=True)

print("\nPLY elements:")
for element in ply.elements:
    print(f"  {element.name}: {element.count:,}")

vertex = ply["vertex"]
names = vertex.data.dtype.names

print("\nVertex properties:")
print(names)

if all(k in names for k in ("x", "y", "z")):
    for axis in ("x", "y", "z"):
        values = np.asarray(vertex[axis])
        finite = values[np.isfinite(values)]

        if len(finite):
            print(
                f"{axis}: min={finite.min():.3f}, "
                f"max={finite.max():.3f}, "
                f"median={np.median(finite):.3f}"
            )

if {"f_dc_0", "opacity", "scale_0", "rot_0"}.issubset(names):
    print("\nDetected: Likely 3D Gaussian Splatting PLY")

elif {"red", "green", "blue"}.issubset(names):
    print("\nDetected: Likely conventional XYZRGB point cloud")

else:
    print("\nRepresentation unknown. Inspect properties.")
