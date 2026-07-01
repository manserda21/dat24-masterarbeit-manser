# src/dataset/create_manual_dataset.py

import argparse
import shutil
from pathlib import Path


def load_frame_list(txt_file: Path):

    with open(txt_file, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def copy_samples(
    frame_names,
    image_dir,
    label_dir,
    output_image_dir,
    output_label_dir,
):

    copied = 0

    for frame_name in frame_names:

        image_path = image_dir / frame_name
        label_path = label_dir / frame_name.replace(".png", ".txt")

        if not image_path.exists():
            print(f"Missing image: {image_path}")
            continue

        # Bild immer kopieren
        shutil.copy2(
            image_path,
            output_image_dir / image_path.name,
        )

        # Label nur kopieren, wenn vorhanden
        if label_path.exists():

            shutil.copy2(
                label_path,
                output_label_dir / label_path.name,
            )

        else:
            print(f"Missing label: {label_path}")

        copied += 1

    return copied


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--h1-dir", required=True)
    parser.add_argument("--h2-dir", required=True)

    parser.add_argument("--h1-list", required=True)
    parser.add_argument("--h2-list", required=True)

    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    output_dir = Path(args.output)

    image_out = output_dir / "images" / "train"
    label_out = output_dir / "labels" / "train"

    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    h1_frames = load_frame_list(Path(args.h1_list))
    h2_frames = load_frame_list(Path(args.h2_list))

    copied_h1 = copy_samples(
        h1_frames,
        Path(args.h1_dir) / "images" / "train",
        Path(args.h1_dir) / "labels" / "train",
        image_out,
        label_out,
    )

    copied_h2 = copy_samples(
        h2_frames,
        Path(args.h2_dir) / "images" / "train",
        Path(args.h2_dir) / "labels" / "train",
        image_out,
        label_out,
    )

    print()
    print("================================")
    print("DONE")
    print("================================")
    print(f"H1 copied: {copied_h1}")
    print(f"H2 copied: {copied_h2}")
    print(f"Total:     {copied_h1 + copied_h2}")


if __name__ == "__main__":
    main()