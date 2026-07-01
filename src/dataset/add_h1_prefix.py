# src/add_h1_prefix.py

import argparse
from pathlib import Path


def rename_files(directory: str):

    directory = Path(directory)

    renamed = 0
    skipped = 0

    for file_path in sorted(directory.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.name.startswith("h1_"):
            skipped += 1
            continue

        new_path = file_path.parent / f"h1_{file_path.name}"

        file_path.rename(new_path)

        renamed += 1

    print("\n====================")
    print("PREFIX ADDED")
    print("====================")
    print(f"Renamed: {renamed}")
    print(f"Skipped: {skipped}")
    print(f"Directory: {directory}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        required=True,
        help="Directory containing files to rename",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    rename_files(args.directory)


if __name__ == "__main__":
    main()