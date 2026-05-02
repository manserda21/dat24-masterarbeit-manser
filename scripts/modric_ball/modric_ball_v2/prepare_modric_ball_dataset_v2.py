from pathlib import Path
import random
import shutil

# =========================
# KONFIGURATION
# =========================
RANDOM_SEED = 42
VAL_RATIO = 0.2

RAW_DATASET_DIR = Path("/data/manser/datasets/modric_ball/modric_ball_v2")
OUTPUT_DIR = Path("/data/manser/datasets/modric_ball_dataset_v2_processed")

IMAGE_SUFFIX = ".png"


# =========================
# RESET OUTPUT
# =========================
def reset_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    (output_dir / "images/train").mkdir(parents=True)
    (output_dir / "images/val").mkdir(parents=True)
    (output_dir / "labels/train").mkdir(parents=True)
    (output_dir / "labels/val").mkdir(parents=True)


# =========================
# SAMPLES SAMMELN
# =========================
def collect_samples(raw_dir: Path):
    images_dir = raw_dir / "images" / "train"
    labels_dir = raw_dir / "labels" / "train"

    samples = []

    for label_path in labels_dir.glob("*.txt"):
        if not label_path.read_text().strip():
            continue  # skip leere labels

        image_path = images_dir / f"{label_path.stem}{IMAGE_SUFFIX}"

        if image_path.exists():
            samples.append((image_path, label_path))

    return samples


# =========================
# SPLIT
# =========================
def split_samples(samples):
    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n_val = int(len(samples) * VAL_RATIO)

    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    return train_samples, val_samples


# =========================
# COPY
# =========================
def copy_samples(samples, output_dir, split):
    for img_path, label_path in samples:
        shutil.copy2(img_path, output_dir / f"images/{split}/{img_path.name}")
        shutil.copy2(label_path, output_dir / f"labels/{split}/{label_path.name}")


# =========================
# DATA.YAML
# =========================
def write_yaml(output_dir: Path):
    yaml_path = output_dir / "data.yaml"

    yaml_content = f"""
path: {output_dir}
train: images/train
val: images/val

names:
  0: modric
  1: ball
"""

    yaml_path.write_text(yaml_content.strip())


# =========================
# MAIN
# =========================
def main():
    print("Erstelle Dataset...")

    reset_output_dir(OUTPUT_DIR)

    samples = collect_samples(RAW_DATASET_DIR)

    print(f"Samples gefunden: {len(samples)}")

    train_samples, val_samples = split_samples(samples)

    copy_samples(train_samples, OUTPUT_DIR, "train")
    copy_samples(val_samples, OUTPUT_DIR, "val")

    write_yaml(OUTPUT_DIR)

    print("\nFertig!")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Dataset: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()