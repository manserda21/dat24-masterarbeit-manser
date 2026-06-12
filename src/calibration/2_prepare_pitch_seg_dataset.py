import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


LINE_CLASSES = [
    "Side line top",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Middle line",
    "Big rect. left top",
    "Big rect. left bottom",
    "Big rect. left main",
    "Small rect. left top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Big rect. right top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Small rect. right top",
    "Small rect. right bottom",
    "Small rect. right main",
]


CLASS_TO_ID = {name: idx for idx, name in enumerate(LINE_CLASSES)}


def mask_to_yolo_polygons(mask, width, height, min_area=20):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        coords = []

        for point in approx.reshape(-1, 2):
            x = float(point[0]) / width
            y = float(point[1]) / height
            coords.extend([x, y])

        if len(coords) >= 6:
            polygons.append(coords)

    return polygons


def create_line_mask(width, height, points, thickness):
    mask = np.zeros((height, width), dtype=np.uint8)

    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        mask,
        [pts],
        isClosed=False,
        color=255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    return mask


def write_label_file(record, label_path, thickness):
    width = int(record["width"])
    height = int(record["height"])

    rows = []

    for line_label, instances in record["lines"].items():
        if line_label not in CLASS_TO_ID:
            continue

        class_id = CLASS_TO_ID[line_label]

        for instance in instances:
            image_points = instance["image"]

            if len(image_points) < 2:
                continue

            mask = create_line_mask(
                width=width,
                height=height,
                points=image_points,
                thickness=thickness,
            )

            polygons = mask_to_yolo_polygons(
                mask=mask,
                width=width,
                height=height,
            )

            for polygon in polygons:
                values = [str(class_id)]
                values.extend([f"{value:.6f}" for value in polygon])
                rows.append(" ".join(values))

    with open(label_path, "w") as file:
        file.write("\n".join(rows))


def prepare_dataset(json_path, images_dir, output_dir, val_ratio, seed, thickness):
    random.seed(seed)

    with open(json_path, "r") as file:
        records = json.load(file)

    records = [
        record
        for record in records
        if len(record.get("lines", {})) > 0
    ]

    random.shuffle(records)

    n_val = max(1, int(len(records) * val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
    ]:
        image_out_dir = output_dir / "images" / split_name
        label_out_dir = output_dir / "labels" / split_name

        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for record in split_records:
            image_name = record["image"]

            src_image = images_dir / image_name
            dst_image = image_out_dir / image_name
            dst_label = label_out_dir / f"{Path(image_name).stem}.txt"

            if not src_image.exists():
                print(f"Missing image: {src_image}")
                continue

            shutil.copy2(src_image, dst_image)

            write_label_file(
                record=record,
                label_path=dst_label,
                thickness=thickness,
            )

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(LINE_CLASSES)},
    }

    with open(output_dir / "data.yaml", "w") as file:
        yaml.safe_dump(data_yaml, file, sort_keys=False)

    print()
    print("YOLO segmentation dataset created")
    print(f"Output: {output_dir}")
    print(f"Train images: {len(train_records)}")
    print(f"Val images: {len(val_records)}")
    print(f"Classes: {len(LINE_CLASSES)}")
    print(f"data.yaml: {output_dir / 'data.yaml'}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thickness", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()

    prepare_dataset(
        json_path=Path(args.json),
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        thickness=args.thickness,
    )


if __name__ == "__main__":
    main()