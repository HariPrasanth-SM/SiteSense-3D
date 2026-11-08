from ultralytics import FastSAM
from pathlib import Path
import numpy as np
import cv2

image_path = "outputs/renders/view_A_rgb.png"
output_dir = Path("outputs/masks")
output_dir.mkdir(parents=True, exist_ok=True)

model = FastSAM("FastSAM-s.pt")

results = model(
    image_path,
    device=0,
    imgsz=640,
    conf=0.35,
    retina_masks=False,
)

result = results[0]

result.save(
    filename=str(output_dir / "view_A_segmented.png")
)

if result.masks is not None:
    masks = result.masks.data.cpu().numpy()

    print(f"Detected {len(masks)} masks")

    for i, mask in enumerate(masks):
        mask = (
            mask.astype(np.uint8) * 255
        )

        if mask.shape[:2] != (480, 640):
            mask = cv2.resize(
                mask,
                (640, 480),
                interpolation=cv2.INTER_NEAREST
            )

        cv2.imwrite(
            str(output_dir / f"mask_{i:03d}.png"),
            mask
        )

else:
    print("No masks detected")

