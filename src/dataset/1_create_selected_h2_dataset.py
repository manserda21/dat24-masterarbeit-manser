# src/create_selected_h2_dataset.py

import argparse
import shutil
from pathlib import Path


def load_selected_frames(txt_file: Path) -> set[str]:

    with open(txt_file, "r") as f:

        return {
            line.strip()
            for line in f
            if line.strip()
        }


def create_dataset(
    images_dir: str,
    labels_dir: str,
    selected_frames_file: str,
    output_dir: str,
):

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    selected_frames_file = Path(
        selected_frames_file
    )

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

    selected_frames = load_selected_frames(
        selected_frames_file
    )

    copied_images = 0
    copied_labels = 0

    for image_name in sorted(selected_frames):

        image_path = images_dir / image_name

        if not image_path.exists():

            print(
                f"WARNING: Missing image: "
                f"{image_name}"
            )
            continue

        shutil.copy2(
            image_path,
            output_images / image_name,
        )

        copied_images += 1

        label_name = (
            Path(image_name)
            .with_suffix(".txt")
            .name
        )

        label_path = (
            labels_dir / label_name
        )

        if label_path.exists():

            shutil.copy2(
                label_path,
                output_labels / label_name,
            )

            copied_labels += 1

        else:

            print(
                f"WARNING: Missing label: "
                f"{label_name}"
            )

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(
        f"Images copied: "
        f"{copied_images}"
    )

    print(
        f"Labels copied: "
        f"{copied_labels}"
    )

    print(
        f"\nOutput directory:\n"
        f"{output_dir}"
    )

    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images-dir",
        required=True,
    )

    parser.add_argument(
        "--labels-dir",
        required=True,
    )

    parser.add_argument(
        "--selected-frames",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    create_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        selected_frames_file=args.selected_frames,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()