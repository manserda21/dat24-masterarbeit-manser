# src/calibration/visualize_calibration_annotations.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_annotations(
    json_path: Path,
):
    with open(
        json_path,
        "r",
    ) as f:
        return json.load(f)


def normalized_to_pixel(
    points,
    width,
    height,
):
    pixel_points = []

    for point in points:

        x = int(
            round(
                point["x"] * width
            )
        )

        y = int(
            round(
                point["y"] * height
            )
        )

        pixel_points.append(
            [x, y]
        )

    return np.array(
        pixel_points,
        dtype=np.int32,
    )


def draw_annotations(
    image,
    annotations,
):
    output = image.copy()

    rng = np.random.default_rng(
        seed=42
    )

    for line_name, points in annotations.items():

        if len(points) < 2:
            continue

        color = tuple(
            int(v)
            for v in rng.integers(
                0,
                255,
                size=3,
            )
        )

        polyline = normalized_to_pixel(
            points,
            width=image.shape[1],
            height=image.shape[0],
        )

        cv2.polylines(
            output,
            [polyline],
            isClosed=False,
            color=color,
            thickness=3,
        )

        label_position = tuple(
            polyline[0]
        )

        cv2.putText(
            output,
            line_name,
            (
                label_position[0] + 5,
                label_position[1] - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def visualize_frame(
    image_path: Path,
    json_path: Path,
    output_path: Path,
):
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    annotations = load_annotations(
        json_path
    )

    vis = draw_annotations(
        image,
        annotations,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        vis,
    )

    print(
        f"Saved: {output_path}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Input image",
    )

    parser.add_argument(
        "--annotation",
        required=True,
        help="SoccerNet json annotation",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output visualization",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    visualize_frame(
        image_path=Path(args.image),
        json_path=Path(args.annotation),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()