from pathlib import Path

import numpy as np
from PIL import Image
import yaml

OUT = Path("outputs/ros")
OUT.mkdir(parents=True, exist_ok=True)

occupancy = np.load(
    "outputs/maps/candidate_occupancy.npy"
)

SCALE = 1.43
RESOLUTION_SCENE = 0.05

X_MIN = -5.5
Z_MIN = -4.5

# ROS map image:
# white = free
# black = occupied
# gray = unknown
image = np.full(
    occupancy.shape,
    205,
    dtype=np.uint8
)

image[occupancy == 0] = 254
image[occupancy == 100] = 0

# Grid row 0 corresponds to minimum Z,
# while image row 0 is displayed at the top.
image = np.flipud(image)

Image.fromarray(image).save(
    OUT / "valmet_map.pgm"
)

metadata = {
    "image": "valmet_map.pgm",
    "mode": "trinary",
    "resolution": RESOLUTION_SCENE * SCALE,
    "origin": [
        X_MIN * SCALE,
        Z_MIN * SCALE,
        0.0
    ],
    "negate": 0,
    "occupied_thresh": 0.65,
    "free_thresh": 0.196
}

with open(OUT / "valmet_map.yaml", "w") as f:
    yaml.safe_dump(metadata, f, sort_keys=False)

print("Saved ROS map:")
print(OUT / "valmet_map.pgm")
print(OUT / "valmet_map.yaml")

