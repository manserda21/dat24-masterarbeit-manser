# src/calibration/visualization/visualize_random_calibration_samples.py

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def load_annotation(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


def normalized_to_pixel(points, width, height):
    return np.array(
        [
            [
                int(round(p["x"] * width)),
                int(round(p["y"] * height)),
            ]
            for p in points
        ],
        dtype=np.int32,
    )


def draw_annotations(image, annotations):
    output = image.copy()

    rng = np.random.default_rng(seed=42)

    for line_name, points in annotations.items():

        if len(points) < 2:
            continue

        color = tuple(
            int(x)
            for x in rng.integers(
                0,
                255,
                size=3,
            )
        )

        polyline = normalized_to_pixel(
            points,
            image.shape[1],
            image.shape[0],
        )

        cv2.polylines(
            output,
            [polyline],
            isClosed=False,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA,
        )

        for point in polyline:
            cv2.circle(
                output,
                tuple(point),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )

        x, y = polyline[0]

        cv2.putText(
            output,
            line_name,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory containing images/ and annotations/",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where visualizations will be stored.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of random samples to visualize.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)

    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(images_dir.glob("*.jpg"))

    random.seed(42)

    samples = random.sample(
        image_files,
        min(
            args.num_samples,
            len(image_files),
        ),
    )

    print(f"Visualizing {len(samples)} images...")

    for image_path in samples:

        annotation_path = (
            annotations_dir
            / f"{image_path.stem}.json"
        )

        if not annotation_path.exists():
            print(f"Missing annotation: {annotation_path.name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not load image: {image_path.name}")
            continue

        annotations = load_annotation(annotation_path)

        visualization = draw_annotations(
            image,
            annotations,
        )

        output_path = (
            output_dir
            / f"{image_path.stem}_annotations.png"
        )

        cv2.imwrite(
            str(output_path),
            visualization,
        )

    print("\nDone.")
    print(f"Saved visualizations to:\n{output_dir}")


if __name__ == "__main__":
    main()