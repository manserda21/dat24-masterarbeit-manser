import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
SCALE = 10.0


def compute_reprojection_error(
    H,
    image_points,
    world_points,
):
    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2).astype(np.float32),
        H,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - world_points,
        axis=1,
    )

    return (
        float(errors.mean()),
        float(errors.max()),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="calibration_dataset.json",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-mean-error",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    homographies = {}
    quality_rows = []

    for record in dataset:

        frame_name = record["image"]

        image_points = []
        world_points = []

        for _, point_data in record["all_points"].items():

            image_points.append(
                point_data["image"]
            )

            world_points.append(
                point_data["world"]
            )

        image_points = np.array(
            image_points,
            dtype=np.float32,
        )

        world_points = (
            np.array(
                world_points,
                dtype=np.float32,
            )
            * SCALE
        )

        if len(image_points) < 4:

            quality_rows.append(
                {
                    "frame": frame_name,
                    "status": "too_few_points",
                    "num_points": len(image_points),
                }
            )

            print(
                f"{frame_name}"
                f" | skipped (only {len(image_points)} points)"
            )

            continue

        H, mask = cv2.findHomography(
            image_points,
            world_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=args.ransac_threshold,
        )

        if H is None:

            quality_rows.append(
                {
                    "frame": frame_name,
                    "status": "homography_failed",
                    "num_points": len(image_points),
                }
            )

            print(
                f"{frame_name}"
                f" | homography failed"
            )

            continue

        mean_error, max_error = compute_reprojection_error(
            H,
            image_points,
            world_points,
        )

        inliers = (
            int(mask.sum())
            if mask is not None
            else len(image_points)
        )

        is_valid = (
            len(image_points) >= args.min_points
            and mean_error < args.max_mean_error
        )

        if is_valid:
            homographies[frame_name] = H
            status = "kept"
        else:
            status = "filtered"

        quality_rows.append(
            {
                "frame": frame_name,
                "status": status,
                "num_points": len(image_points),
                "num_inliers": inliers,
                "mean_error_px": mean_error,
                "max_error_px": max_error,
            }
        )

        print(
            f"{frame_name}"
            f" | status={status}"
            f" | points={len(image_points)}"
            f" | inliers={inliers}"
            f" | mean={mean_error:.2f}"
            f" | max={max_error:.2f}"
        )

    np.savez(
        output_dir / "homographies.npz",
        **homographies,
    )

    pd.DataFrame(
        quality_rows
    ).to_csv(
        output_dir / "homography_quality.csv",
        index=False,
    )

    print()
    print("========================")
    print("Finished")
    print("========================")
    print(f"Total frames: {len(dataset)}")
    print(f"Stored homographies: {len(homographies)}")
    print(f"Filtered frames: {len(dataset) - len(homographies)}")
    print()

    print(
        f"Homographies: "
        f"{output_dir / 'homographies.npz'}"
    )

    print(
        f"Quality CSV: "
        f"{output_dir / 'homography_quality.csv'}"
    )


if __name__ == "__main__":
    main()