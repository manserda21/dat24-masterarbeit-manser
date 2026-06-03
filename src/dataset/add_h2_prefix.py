# src/add_h2_prefix.py

import argparse
from pathlib import Path


def rename_files(directory: str):

    directory = Path(directory)

    files = sorted(directory.iterdir())

    renamed = 0

    for file_path in files:

        if not file_path.is_file():
            continue

        if file_path.name.startswith("h2_"):
            continue

        new_path = (
            file_path.parent
            / f"h2_{file_path.name}"
        )

        file_path.rename(new_path)

        renamed += 1

    print(
        f"Renamed {renamed} files in:\n"
        f"{directory}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--directory",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    rename_files(args.directory)


if __name__ == "__main__":
    main()