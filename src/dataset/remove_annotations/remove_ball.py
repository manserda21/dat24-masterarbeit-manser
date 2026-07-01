from pathlib import Path
import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="Remove ball annotations (class 11) from YOLO labels"
    )

    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Path to label directory"
    )

    parser.add_argument(
        "--ball-class",
        type=int,
        default=11,
        help="Class ID of the ball"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    labels_dir = Path(args.labels_dir)

    removed = 0

    for txt_file in labels_dir.glob("*.txt"):

        with open(txt_file, "r") as f:
            lines = f.readlines()

        filtered_lines = []

        for line in lines:

            if line.startswith(f"{args.ball_class} "):
                removed += 1
                continue

            filtered_lines.append(line)

        with open(txt_file, "w") as f:
            f.writelines(filtered_lines)

    print(f"Removed {removed} annotations.")


if __name__ == "__main__":
    main()