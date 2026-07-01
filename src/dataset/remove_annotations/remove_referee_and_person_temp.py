# src/remove_referee_and_person_temp.py

import argparse
from pathlib import Path


def process_label_file(label_file: Path):

    new_lines = []

    with open(label_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            class_id = int(parts[0])

            # ----------------------------------
            # REMOVE REFEREE
            # ----------------------------------

            if class_id == 11:
                continue

            # ----------------------------------
            # REMOVE PERSON_TEMP
            # ----------------------------------

            if class_id == 12:
                continue

            # ----------------------------------
            # BALL: 13 -> 11
            # ----------------------------------

            if class_id == 13:
                parts[0] = "11"

            new_lines.append(
                " ".join(parts)
            )

    with open(label_file, "w") as f:

        for line in new_lines:
            f.write(line + "\n")


def run(labels_dir: str):

    labels_dir = Path(labels_dir)

    if not labels_dir.exists():

        raise FileNotFoundError(
            f"Directory not found:\n{labels_dir}"
        )

    label_files = sorted(
        labels_dir.rglob("*.txt")
    )

    print("\n==============================")
    print("UPDATING LABEL FILES")
    print("==============================")

    print(
        f"Found {len(label_files)} label files"
    )

    for label_file in label_files:
        process_label_file(label_file)

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(
        f"Processed {len(label_files)} files"
    )

    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing label files",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run(
        labels_dir=args.labels_dir,
    )


if __name__ == "__main__":
    main()