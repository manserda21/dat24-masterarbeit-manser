from pathlib import Path
import argparse
import re


REMOVE_RANGES = [
    (10550, 11550),  # Split-Screen Barcelona
    (21800, 22300),  # Split-Screen Barcelona
]


def should_remove(frame_number: int) -> bool:
    return any(start <= frame_number <= end for start, end in REMOVE_RANGES)


def clean_dataset(dataset_dir: Path):
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    train_txt = dataset_dir / "train.txt"

    removed_images = 0
    removed_labels = 0

    # Bilder entfernen
    for image_file in images_dir.glob("h1_frame_*.png"):
        frame = int(image_file.stem.replace("h1_frame_", ""))

        if should_remove(frame):
            image_file.unlink()
            removed_images += 1

    # Labels entfernen
    for label_file in labels_dir.glob("h1_frame_*.txt"):
        frame = int(label_file.stem.replace("h1_frame_", ""))

        if should_remove(frame):
            label_file.unlink()
            removed_labels += 1

    # train.txt aktualisieren
    with open(train_txt, "r") as f:
        lines = f.readlines()

    filtered_lines = []

    for line in lines:
        match = re.search(r"h1_frame_(\d+)\.png", line)

        if match:
            frame = int(match.group(1))

            if should_remove(frame):
                continue

        filtered_lines.append(line)

    with open(train_txt, "w") as f:
        f.writelines(filtered_lines)

    print(f"Entfernte Bilder: {removed_images}")
    print(f"Entfernte Labels: {removed_labels}")
    print(f"Verbleibende train.txt Einträge: {len(filtered_lines)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Pfad zum YOLO-Dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()