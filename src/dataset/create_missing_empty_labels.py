# src/create_missing_empty_labels.py

import argparse
from pathlib import Path


def run(images_dir, labels_dir):

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    created = 0

    for image_path in images_dir.glob("*"):

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            label_path.touch()

            created += 1

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(
        f"Created empty labels: "
        f"{created}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-dir",
        required=True,
    )

    parser.add_argument(
        "--labels-dir",
        required=True,
    )

    args = parser.parse_args()

    run(
        args.images_dir,
        args.labels_dir,
    )


if __name__ == "__main__":
    main()