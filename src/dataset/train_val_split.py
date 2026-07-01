# src/train_val_split.py

import argparse
import random
import shutil
from pathlib import Path


CLASS_NAMES = [
    "opponent",
    "danilo",
    "varane",
    "ramos",
    "marcelo",
    "modric",
    "casemiro",
    "kroos",
    "isco",
    "benzema",
    "ronaldo",
]


def reset_output_dir(output_dir: Path):

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:

        (output_dir / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (output_dir / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


def collect_samples(
    image_dir: Path,
    label_dir: Path,
):

    image_paths = sorted(
        image_dir.glob("*.png")
    )

    samples = []

    for image_path in image_paths:

        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():

            print(
                f"WARNING: Missing label "
                f"for {image_path.name}"
            )
            continue

        samples.append(
            (image_path, label_path)
        )

    return samples


def split_samples(
    samples,
    train_ratio: float,
    random_seed: int,
):

    random.seed(random_seed)

    random.shuffle(samples)

    n_train = int(
        len(samples) * train_ratio
    )

    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    return train_samples, val_samples


def copy_samples(
    samples,
    split: str,
    output_dir: Path,
):

    for image_path, label_path in samples:

        shutil.copy2(
            image_path,
            output_dir
            / "images"
            / split
            / image_path.name,
        )

        shutil.copy2(
            label_path,
            output_dir
            / "labels"
            / split
            / label_path.name,
        )


def create_yaml(
    output_dir: Path,
):

    yaml_lines = [
        f"path: {output_dir}",
        "",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    for idx, class_name in enumerate(
        CLASS_NAMES
    ):
        yaml_lines.append(
            f"  {idx}: {class_name}"
        )

    yaml_content = "\n".join(
        yaml_lines
    )

    (
        output_dir / "data.yaml"
    ).write_text(yaml_content)


def run_split(
    source_dir: str,
    output_dir: str,
    train_ratio: float,
    random_seed: int,
):

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    image_dir = (
        source_dir
        / "images"
        / "train"
    )

    label_dir = (
        source_dir
        / "labels"
        / "train"
    )

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{image_dir}"
        )

    if not label_dir.exists():

        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"{label_dir}"
        )

    reset_output_dir(output_dir)

    samples = collect_samples(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    train_samples, val_samples = (
        split_samples(
            samples=samples,
            train_ratio=train_ratio,
            random_seed=random_seed,
        )
    )

    copy_samples(
        train_samples,
        "train",
        output_dir,
    )

    copy_samples(
        val_samples,
        "val",
        output_dir,
    )

    create_yaml(
        output_dir,
    )

    print("\n==============================")
    print("DATASET SPLIT FINISHED")
    print("==============================")

    print(
        f"Total samples: "
        f"{len(samples)}"
    )

    print(
        f"Train samples: "
        f"{len(train_samples)}"
    )

    print(
        f"Val samples: "
        f"{len(val_samples)}"
    )

    print(
        f"\nOutput:\n"
        f"{output_dir}"
    )

    print()


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create train/val split "
            "for YOLO dataset."
        )
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help=(
            "Dataset directory "
            "(contains images/train "
            "and labels/train)"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output split dataset "
            "directory"
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run_split(
        source_dir=args.source,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()