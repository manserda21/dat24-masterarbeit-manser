# src/prepare_cvat_export.py

import argparse
import shutil
import re
from pathlib import Path


def extract_frame_number(filename: str) -> int:
    """
    Extract frame number from filename.

    Example:
    frame_030600.png -> 30600
    """

    match = re.search(r"frame_(\d+)", filename)

    if match is None:
        raise ValueError(
            f"Could not extract frame number from {filename}"
        )

    return int(match.group(1))


def prepare_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    prefix: str,
    max_frame: int | None = None,
):
    """
    Copies images and labels to a new dataset,
    adds prefix and optionally filters by frame number.
    """

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    output_dir = Path(output_dir)

    output_images = output_dir / "images"
    output_labels = output_dir / "labels"

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    copied = 0

    for image_path in image_files:

        frame_number = extract_frame_number(
            image_path.name
        )

        if (
            max_frame is not None
            and frame_number > max_frame
        ):
            continue

        new_image_name = (
            f"{prefix}{image_path.name}"
        )

        shutil.copy2(
            image_path,
            output_images / new_image_name,
        )

        label_path = (
            labels_dir /
            image_path.with_suffix(".txt").name
        )

        if label_path.exists():

            new_label_name = (
                f"{prefix}{label_path.name}"
            )

            shutil.copy2(
                label_path,
                output_labels / new_label_name,
            )

        copied += 1

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(f"Copied images: {copied}")
    print(f"Output: {output_dir}")
    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    prepare_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        prefix=args.prefix,
        max_frame=args.max_frame,
    )


if __name__ == "__main__":
    main()