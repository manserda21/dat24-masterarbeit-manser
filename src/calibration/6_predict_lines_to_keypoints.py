import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "/data/manser/runs/pitch_calibration/pitch_lines_v1/weights/best.pt"

IMAGE_PATH = "/data/manser/datasets/calibration/malaga_real_2016_pitch_calibration_dataset_points_lines_v4/images/frame_001298.PNG"


def fit_line_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) < 20:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    )

    return {
        "vx": float(vx),
        "vy": float(vy),
        "x0": float(x0),
        "y0": float(y0),
    }


def main():

    model = YOLO(MODEL_PATH)

    results = model.predict(
        IMAGE_PATH,
        conf=0.5,
        verbose=False,
    )

    result = results[0]

    if result.masks is None:
        print("No masks detected")
        return

    image = cv2.imread(IMAGE_PATH)

    for mask, cls_id in zip(
        result.masks.data.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
    ):

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        line = fit_line_from_mask(mask)

        if line is None:
            continue

        print(cls_id, line)


if __name__ == "__main__":
    main()