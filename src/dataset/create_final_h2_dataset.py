# src/create_final_h2_dataset.py

import argparse
import shutil
from pathlib import Path


# =========================================================
# COPY FILES
# =========================================================

def copy_files(
    source_dir: Path,
    output_dir: Path,
    pattern: str,
):

    copied = 0

    for file_path in sorted(
        source_dir.glob(pattern)
    ):

        destination = (
            output_dir / file_path.name
        )

        if destination.exists():

            print(
                f"WARNING: File already exists: "
                f"{file_path.name}"
            )

            continue

        shutil.copy2(
            file_path,
            destination,
        )

        copied += 1

    return copied


# =========================================================
# CREATE DATA.YAML
# =========================================================

def create_data_yaml(
    output_dir: Path,
):

    yaml_content = """
path: .

train: images
val: images

names:
  0: opponent
  1: danilo
  2: varane
  3: ramos
  4: marcelo
  5: modric
  6: casemiro
  7: kroos
  8: isco
  9: benzema
  10: ronaldo
  11: ball
"""

    (
        output_dir / "data.yaml"
    ).write_text(
        yaml_content.strip()
    )


# =========================================================
# MAIN FUNCTION
# =========================================================

def create_dataset(
    manual_images_dir: str,
    manual_labels_dir: str,
    prediction_images_dir: str,
    prediction_labels_dir: str,
    output_dir: str,
):

    manual_images_dir = Path(
        manual_images_dir
    )

    manual_labels_dir = Path(
        manual_labels_dir
    )

    prediction_images_dir = Path(
        prediction_images_dir
    )

    prediction_labels_dir = Path(
        prediction_labels_dir
    )

    output_dir = Path(output_dir)

    output_images = (
        output_dir / "images"
    )

    output_labels = (
        output_dir / "labels"
    )

    # -----------------------------------------------------
    # RESET OUTPUT
    # -----------------------------------------------------

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # COPY MANUAL H2 DATA
    # -----------------------------------------------------

    manual_images = copy_files(
        manual_images_dir,
        output_images,
        "h2_*.png",
    )

    manual_labels = copy_files(
        manual_labels_dir,
        output_labels,
        "h2_*.txt",
    )

    # -----------------------------------------------------
    # COPY PREDICTION DATA
    # -----------------------------------------------------

    prediction_images = copy_files(
        prediction_images_dir,
        output_images,
        "h2_*.png",
    )

    prediction_labels = copy_files(
        prediction_labels_dir,
        output_labels,
        "h2_*.txt",
    )

    # -----------------------------------------------------
    # DATA YAML
    # -----------------------------------------------------

    create_data_yaml(
        output_dir
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    total_images = len(
        list(output_images.glob("*.png"))
    )

    total_labels = len(
        list(output_labels.glob("*.txt"))
    )

    print("\n==============================")
    print("FINAL H2 DATASET CREATED")
    print("==============================")

    print(
        f"Manual images:      "
        f"{manual_images}"
    )

    print(
        f"Manual labels:      "
        f"{manual_labels}"
    )

    print(
        f"Prediction images:  "
        f"{prediction_images}"
    )

    print(
        f"Prediction labels:  "
        f"{prediction_labels}"
    )

    print()

    print(
        f"Total images: "
        f"{total_images}"
    )

    print(
        f"Total labels: "
        f"{total_labels}"
    )

    print(
        f"\nOutput:\n"
        f"{output_dir}"
    )

    print()


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manual-images",
        required=True,
    )

    parser.add_argument(
        "--manual-labels",
        required=True,
    )

    parser.add_argument(
        "--prediction-images",
        required=True,
    )

    parser.add_argument(
        "--prediction-labels",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================

def main():

    args = parse_args()

    create_dataset(
        manual_images_dir=args.manual_images,
        manual_labels_dir=args.manual_labels,
        prediction_images_dir=args.prediction_images,
        prediction_labels_dir=args.prediction_labels,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()