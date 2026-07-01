from pathlib import Path
import shutil
import argparse


BALL_CLASS_ID = 11


def process_labels(label_file):
    ball_lines = []

    with open(label_file, "r") as f:
        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            if class_id == BALL_CLASS_ID:
                parts[0] = "0"
                ball_lines.append(" ".join(parts))

    return ball_lines


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    image_src = input_dir / "images" / "train"
    label_src = input_dir / "labels" / "train"

    image_dst = output_dir / "images" / "train"
    label_dst = output_dir / "labels" / "train"

    image_dst.mkdir(parents=True, exist_ok=True)
    label_dst.mkdir(parents=True, exist_ok=True)

    image_count = 0
    ball_frames = 0
    ball_annotations = 0

    for image_file in sorted(image_src.glob("*.png")):

        image_count += 1

        shutil.copy2(
            image_file,
            image_dst / image_file.name,
        )

        label_file = label_src / f"{image_file.stem}.txt"

        output_label = label_dst / f"{image_file.stem}.txt"

        if not label_file.exists():
            output_label.touch()
            continue

        ball_lines = process_labels(label_file)

        if ball_lines:
            ball_frames += 1
            ball_annotations += len(ball_lines)

        with open(output_label, "w") as f:
            for line in ball_lines:
                f.write(line + "\n")

    data_yaml = output_dir / "data.yaml"

    with open(data_yaml, "w") as f:
        f.write(
            "path: .\n"
            "train: images/train\n"
            "\n"
            "nc: 1\n"
            "\n"
            "names:\n"
            "  0: ball\n"
        )

    print("\n========== DONE ==========")
    print(f"Images: {image_count}")
    print(f"Ball frames: {ball_frames}")
    print(f"Ball annotations: {ball_annotations}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()