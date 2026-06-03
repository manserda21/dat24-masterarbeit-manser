# src.extract_unannotated_h2_frames.py

import argparse
import shutil
from pathlib import Path


def load_selected_frames(
    selected_frames_file: Path,
) -> set[str]:

    selected_frames = set()

    with open(selected_frames_file, "r") as f:

        for line in f:

            filename = line.strip()

            if not filename:
                continue

            if filename.startswith("h2_"):
                filename = filename[3:]

            selected_frames.add(filename)

    return selected_frames

def run(
    raw_dir: str,
    selected_frames_file: str,
    output_dir: str,
):

    raw_dir = Path(raw_dir)

    selected_frames_file = Path(
        selected_frames_file
    )

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_frames = (
        load_selected_frames(
            selected_frames_file
        )
    )

    all_h2_images = sorted(
        raw_dir.glob("*.png")
    )

    copied = 0

    for image_path in all_h2_images:

        if image_path.name in selected_frames:
            continue

        shutil.copy2(
            image_path,
            output_dir / image_path.name,
        )

        copied += 1

    print("\n==============================")
    print("UNANNOTATED H2 EXTRACTION")
    print("==============================")

    print(
        f"Annotated H2 frames: "
        f"{len(selected_frames)}"
    )

    print(
        f"Total H2 frames: "
        f"{len(all_h2_images)}"
    )

    print(
        f"Remaining H2 frames: "
        f"{copied}"
    )

    print(
        f"\nOutput:\n{output_dir}"
    )

    print()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        required=True,
    )

    parser.add_argument(
        "--selected-frames",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run(
        raw_dir=args.raw_dir,
        selected_frames_file=args.selected_frames,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()