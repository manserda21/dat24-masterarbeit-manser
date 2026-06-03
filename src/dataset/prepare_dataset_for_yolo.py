from pathlib import Path
import argparse
import random
import shutil


# =========================================================
# DEFAULT CONFIG
# =========================================================

DEFAULT_RANDOM_SEED = 42

DEFAULT_TRAIN_RATIO = 0.8

SUPPORTED_IMAGE_SUFFIXES = [
    ".jpg",
    ".jpeg",
    ".png",
]


# =========================================================
# ARGUMENT PARSER
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Prepare CVAT YOLO export for Ultralytics YOLO training"
    )

    parser.add_argument(
        "--raw_dataset_dir",
        type=str,
        required=True,
        help="Path to raw CVAT YOLO export"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to processed YOLO dataset"
    )

    parser.add_argument(
        "--target_classes",
        type=int,
        nargs="+",
        required=True,
        help="Original class IDs to keep"
    )

    parser.add_argument(
        "--target_class_names",
        type=str,
        nargs="+",
        required=True,
        help="New class names in desired order"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Train split ratio"
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed"
    )

    parser.add_argument(
        "--keep_empty_labels",
        action="store_true",
        help="Keep empty label files"
    )

    return parser.parse_args()


# =========================================================
# RESET OUTPUT DIRECTORY
# =========================================================

def reset_output_dir(output_dir: Path):

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:

        (output_dir / f"images/{split}").mkdir(
            parents=True,
            exist_ok=True
        )

        (output_dir / f"labels/{split}").mkdir(
            parents=True,
            exist_ok=True
        )


# =========================================================
# FIND IMAGE FOR LABEL
# =========================================================

def find_image(images_dir: Path, stem: str):

    for suffix in SUPPORTED_IMAGE_SUFFIXES:

        image_path = images_dir / f"{stem}{suffix}"

        if image_path.exists():
            return image_path

    return None


# =========================================================
# CREATE CLASS MAPPING
# =========================================================

def create_class_mapping(target_classes):

    return {
        original_id: new_id
        for new_id, original_id in enumerate(target_classes)
    }


# =========================================================
# PROCESS LABEL FILE
# =========================================================

def process_label_file(
    label_path: Path,
    target_class_mapping
):

    processed_lines = []

    lines = label_path.read_text().strip().splitlines()

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            continue

        old_class_id = int(parts[0])

        if old_class_id not in target_class_mapping:
            continue

        new_class_id = target_class_mapping[
            old_class_id
        ]

        new_line = " ".join(
            [str(new_class_id)] + parts[1:]
        )

        processed_lines.append(new_line)

    return processed_lines


# =========================================================
# COLLECT SAMPLES
# =========================================================

def collect_samples(
    raw_dataset_dir: Path,
    target_class_mapping,
    keep_empty_labels
):

    images_dir = raw_dataset_dir / "images" / "train"
    labels_dir = raw_dataset_dir / "labels" / "train"

    samples = []

    missing_images = []

    for label_path in sorted(labels_dir.glob("*.txt")):

        processed_lines = process_label_file(
            label_path,
            target_class_mapping
        )

        if (
            not keep_empty_labels
            and not processed_lines
        ):
            continue

        image_path = find_image(
            images_dir,
            label_path.stem
        )

        if image_path is None:
            missing_images.append(label_path.name)
            continue

        samples.append(
            (
                image_path,
                label_path,
                processed_lines,
            )
        )

    return samples, missing_images


# =========================================================
# SPLIT DATASET
# =========================================================

def split_samples(
    samples,
    train_ratio,
    random_seed
):

    random.seed(random_seed)

    random.shuffle(samples)

    n_train = int(len(samples) * train_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    return train_samples, val_samples


# =========================================================
# COPY SAMPLES
# =========================================================

def copy_samples(
    samples,
    output_dir: Path,
    split: str
):

    for image_path, label_path, processed_lines in samples:

        # -------------------------------------------------
        # COPY IMAGE
        # -------------------------------------------------

        shutil.copy2(
            image_path,
            output_dir / f"images/{split}/{image_path.name}"
        )

        # -------------------------------------------------
        # WRITE LABEL
        # -------------------------------------------------

        output_label_path = (
            output_dir
            / f"labels/{split}/{label_path.name}"
        )

        output_label_path.write_text(
            "\n".join(processed_lines)
        )


# =========================================================
# WRITE DATA.YAML
# =========================================================

def write_yaml(
    output_dir: Path,
    class_names
):

    yaml_lines = [
        f"path: {output_dir}",
        "",
        "train: images/train",
        "val: images/val",
        "",
        "names:"
    ]

    for class_id, class_name in enumerate(class_names):

        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    yaml_content = "\n".join(yaml_lines)

    yaml_path = output_dir / "data.yaml"

    yaml_path.write_text(yaml_content)


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    total_samples,
    train_samples,
    val_samples,
    missing_images,
    class_mapping,
    class_names,
    output_dir
):

    print("\n==============================")
    print("DATASET SUMMARY")
    print("==============================")

    print(f"Total samples:  {total_samples}")
    print(f"Train samples:  {len(train_samples)}")
    print(f"Val samples:    {len(val_samples)}")
    print(f"Missing images: {len(missing_images)}")

    print("\nClass mapping:")

    for original_id, new_id in class_mapping.items():

        print(
            f"  Original {original_id}"
            f" -> New {new_id}"
            f" ({class_names[new_id]})"
        )

    print("\nOutput directory:")
    print(output_dir)

    print("\nDone.\n")


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_arguments()

    raw_dataset_dir = Path(args.raw_dataset_dir)

    output_dir = Path(args.output_dir)

    target_class_mapping = create_class_mapping(
        args.target_classes
    )

    print("\nPreparing YOLO dataset...\n")

    reset_output_dir(output_dir)

    samples, missing_images = collect_samples(
        raw_dataset_dir,
        target_class_mapping,
        args.keep_empty_labels
    )

    print(f"Valid samples found: {len(samples)}")

    train_samples, val_samples = split_samples(
        samples,
        args.train_ratio,
        args.random_seed
    )

    copy_samples(
        train_samples,
        output_dir,
        "train"
    )

    copy_samples(
        val_samples,
        output_dir,
        "val"
    )

    write_yaml(
        output_dir,
        args.target_class_names
    )

    print_summary(
        len(samples),
        train_samples,
        val_samples,
        missing_images,
        target_class_mapping,
        args.target_class_names,
        output_dir
    )


if __name__ == "__main__":
    main()