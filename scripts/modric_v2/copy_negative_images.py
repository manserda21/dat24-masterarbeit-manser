from pathlib import Path
import shutil

# ----------------------------
# Konfiguration
# ----------------------------
RAW_DIR = Path("/data/manser/datasets/modric_v2/raw/modric_dataset_v2")
OUTPUT_DIR = Path("/data/manser/datasets/modric_v2/intermediate/negative_samples_review")

TAKE_EVERY_NTH = 15
MAX_CANDIDATES = 300
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main():
    images_dir = RAW_DIR / "images" / "train"
    labels_dir = RAW_DIR / "labels" / "train"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images-Ordner nicht gefunden: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels-Ordner nicht gefunden: {labels_dir}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labeled_stems = {p.stem for p in labels_dir.glob("*.txt")}

    unlabeled_images = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in labeled_stems:
            continue
        unlabeled_images.append(image_path)

    selected = unlabeled_images[::TAKE_EVERY_NTH][:MAX_CANDIDATES]

    for image_path in selected:
        shutil.copy2(image_path, OUTPUT_DIR / image_path.name)

    print(f"Unlabeled Bilder insgesamt: {len(unlabeled_images)}")
    print(f"Kopierte Review-Kandidaten: {len(selected)}")
    print(f"Gespeichert in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()