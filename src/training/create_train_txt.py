# src/create_train_txt.py

import argparse
from pathlib import Path


def create_train_txt(
    images_dir: str,
    output_file: str,
):
    """
    Creates a train.txt file containing one image path per line.

    Example:
    images/train/frame_000000.png
    images/train/frame_000050.png
    ...
    """

    images_dir = Path(images_dir)
    output_file = Path(output_file)

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    with open(output_file, "w") as f:

        for image_path in image_files:

            f.write(
                f"images/train/{image_path.name}\n"
            )

    print("\n==============================")
    print("DONE")
    print("==============================")
    print(f"Images found: {len(image_files)}")
    print(f"Output file: {output_file}")
    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing images",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output train.txt file",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    create_train_txt(
        images_dir=args.images_dir,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()