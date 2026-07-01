from pathlib import Path
import argparse
import random
import shutil


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Ordner mit images/train und labels/train"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Zielordner für den Split"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Anteil Trainingsdaten"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    image_dir = input_dir / "images" / "train"
    label_dir = input_dir / "labels" / "train"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    # Altes Split-Verzeichnis löschen
    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"

    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.png"))

    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * args.train_ratio)

    train_images = images[:n_train]
    val_images = images[n_train:]

    # Train
    for image_file in train_images:

        shutil.copy2(
            image_file,
            train_img_dir / image_file.name
        )

        label_file = label_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing label file: {label_file}"
            )

        shutil.copy2(
            label_file,
            train_lbl_dir / label_file.name
        )

    # Validation
    for image_file in val_images:

        shutil.copy2(
            image_file,
            val_img_dir / image_file.name
        )

        label_file = label_dir / f"{image_file.stem}.txt"

        if not label_file.exists():
            raise FileNotFoundError(
                f"Missing label file: {label_file}"
            )

        shutil.copy2(
            label_file,
            val_lbl_dir / label_file.name
        )

    yaml_text = """path: .
train: images/train
val: images/val

nc: 1

names:
  0: ball
"""

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_text)

    print("\n========== SPLIT DONE ==========")
    print(f"Total images: {n_total}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images:   {len(val_images)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()