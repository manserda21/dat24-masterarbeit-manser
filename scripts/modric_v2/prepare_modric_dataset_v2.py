from pathlib import Path
import random
import shutil

# ----------------------------
# Konfiguration
# ----------------------------
RANDOM_SEED = 42
VAL_RATIO = 0.2

# Positiver CVAT-Export
RAW_POS_DIR = Path("/data/manser/datasets/modric_v2/raw/modric_dataset_v2")

# Ordner mit manuell geprueften Negativbildern
# Hier sollen nur Bilder liegen, in denen Modric sicher NICHT vorkommt.
NEGATIVE_IMAGES_DIR = Path("/data/manser/datasets/modric_v2/intermediate/negative_samples")

# Zielordner fuer das fertige YOLO-Dataset
OUTPUT_DIR = Path("/data/manser/datasets/modric_v2/processed/modric_dataset_v2")

# Erlaubte Bildendungen fuer Negativbilder
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def reset_output_dir(output_dir: Path) -> None:
    """Loescht den Zielordner, falls vorhanden, und erstellt die Struktur neu."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)


def collect_positive_samples(raw_pos_dir: Path) -> list[tuple[Path, Path]]:
    """
    Sammelt nur positive Beispiele:
    - Bild existiert
    - Label-Datei existiert
    - Label-Datei ist nicht leer
    """
    images_dir = raw_pos_dir / "images" / "train"
    labels_dir = raw_pos_dir / "labels" / "train"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images-Ordner nicht gefunden: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels-Ordner nicht gefunden: {labels_dir}")

    samples: list[tuple[Path, Path]] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        content = label_path.read_text().strip()
        if not content:
            continue

        image_path = images_dir / f"{label_path.stem}.png"
        if not image_path.exists():
            continue

        samples.append((image_path, label_path))

    return samples


def collect_negative_samples(negative_images_dir: Path) -> list[tuple[Path, None]]:
    """
    Sammelt Negativbilder:
    - Nur Bilder
    - Labels werden spaeter als leere .txt Dateien erzeugt
    """
    if not negative_images_dir.exists():
        raise FileNotFoundError(f"Negativbilder-Ordner nicht gefunden: {negative_images_dir}")

    samples: list[tuple[Path, None]] = []

    for image_path in sorted(negative_images_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            samples.append((image_path, None))

    return samples


def split_samples(
    positive_samples: list[tuple[Path, Path]],
    negative_samples: list[tuple[Path, None]],
    val_ratio: float,
    random_seed: int,
) -> tuple[list[tuple[Path, Path | None]], list[tuple[Path, Path | None]]]:
    """
    Splittet positive und negative Samples getrennt in train/val,
    damit beide Mengen in beiden Splits vertreten sind.
    """
    random.seed(random_seed)

    pos = positive_samples.copy()
    neg = negative_samples.copy()

    random.shuffle(pos)
    random.shuffle(neg)

    n_val_pos = max(1, int(len(pos) * val_ratio)) if pos else 0
    n_val_neg = max(1, int(len(neg) * val_ratio)) if neg else 0

    val_samples: list[tuple[Path, Path | None]] = []
    train_samples: list[tuple[Path, Path | None]] = []

    val_samples.extend(pos[:n_val_pos])
    train_samples.extend(pos[n_val_pos:])

    val_samples.extend(neg[:n_val_neg])
    train_samples.extend(neg[n_val_neg:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def copy_sample(
    image_path: Path,
    label_path: Path | None,
    output_dir: Path,
    split: str,
) -> None:
    """
    Kopiert Bild und erzeugt/kopiert die passende Label-Datei.
    Bei Negativbeispielen wird eine leere Label-Datei erstellt.
    """
    image_dst = output_dir / "images" / split / image_path.name
    label_dst = output_dir / "labels" / split / f"{image_path.stem}.txt"

    shutil.copy2(image_path, image_dst)

    if label_path is None:
        label_dst.write_text("")
    else:
        shutil.copy2(label_path, label_dst)


def write_data_yaml(output_dir: Path) -> None:
    """Schreibt die data.yaml fuer YOLO."""
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {output_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: modric\n"
    )


def count_files(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern)))


def main() -> None:
    print("Starte Erstellung von modric_dataset_v2 ...")

    reset_output_dir(OUTPUT_DIR)

    positive_samples = collect_positive_samples(RAW_POS_DIR)
    negative_samples = collect_negative_samples(NEGATIVE_IMAGES_DIR)

    print(f"Positive Beispiele gefunden: {len(positive_samples)}")
    print(f"Negative Beispiele gefunden: {len(negative_samples)}")

    train_samples, val_samples = split_samples(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        val_ratio=VAL_RATIO,
        random_seed=RANDOM_SEED,
    )

    for image_path, label_path in train_samples:
        copy_sample(image_path, label_path, OUTPUT_DIR, split="train")

    for image_path, label_path in val_samples:
        copy_sample(image_path, label_path, OUTPUT_DIR, split="val")

    write_data_yaml(OUTPUT_DIR)

    train_images = count_files(OUTPUT_DIR / "images" / "train", "*")
    val_images = count_files(OUTPUT_DIR / "images" / "val", "*")
    train_labels = count_files(OUTPUT_DIR / "labels" / "train", "*.txt")
    val_labels = count_files(OUTPUT_DIR / "labels" / "val", "*.txt")

    print("\nFertig.")
    print(f"Dataset liegt in: {OUTPUT_DIR}")
    print(f"Train images: {train_images}")
    print(f"Val images:   {val_images}")
    print(f"Train labels: {train_labels}")
    print(f"Val labels:   {val_labels}")


if __name__ == "__main__":
    main()