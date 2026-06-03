from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

DATASET_DIR = Path(
    "/data/manser/datasets/modric_kroos_ball_dataset_v1/processed"
)

SPLIT = "train"

NUM_SAMPLES = 10

CLASS_NAMES = {
    0: "modric",
    1: "ball",
    2: "kroos",
}


# =========================================================
# PATHS
# =========================================================

IMAGES_DIR = DATASET_DIR / f"images/{SPLIT}"
LABELS_DIR = DATASET_DIR / f"labels/{SPLIT}"


# =========================================================
# COLORS
# =========================================================

COLORS = {
    0: (0, 255, 0),      # green
    1: (0, 255, 255),    # yellow
    2: (255, 0, 0),      # blue
}


# =========================================================
# YOLO -> XYXY
# =========================================================

def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height
):

    x_center *= image_width
    y_center *= image_height

    width *= image_width
    height *= image_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)

    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return x1, y1, x2, y2


# =========================================================
# DRAW LABELS
# =========================================================

def draw_labels(image, label_path):

    image_height, image_width = image.shape[:2]

    lines = label_path.read_text().strip().splitlines()

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center,
            y_center,
            width,
            height,
            image_width,
            image_height
        )

        color = COLORS.get(class_id, (255, 255, 255))

        class_name = CLASS_NAMES.get(
            class_id,
            str(class_id)
        )

        # -------------------------------------------------
        # DRAW BOX
        # -------------------------------------------------

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # -------------------------------------------------
        # DRAW LABEL
        # -------------------------------------------------

        cv2.putText(
            image,
            class_name,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    return image


# =========================================================
# MAIN
# =========================================================

def main():

    image_paths = list(IMAGES_DIR.glob("*"))

    random.shuffle(image_paths)

    selected_images = image_paths[:NUM_SAMPLES]

    for image_path in selected_images:

        label_path = (
            LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            continue

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        image = cv2.imread(str(image_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # DRAW LABELS
        # -------------------------------------------------

        image = draw_labels(
            image,
            label_path
        )

        # -------------------------------------------------
        # SHOW
        # -------------------------------------------------

        OUTPUT_VIS_DIR = DATASET_DIR / "visualizations"
        OUTPUT_VIS_DIR.mkdir(exist_ok=True)

        output_path = OUTPUT_VIS_DIR / image_path.name

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()