# src/dataset/split_ground_truth_predictions.py

import argparse
import shutil
from pathlib import Path


def load_selected_frames(file_path: Path) -> set[str]:
    with open(file_path, "r") as f:
        return {
            line.strip()
            for line in f
            if line.strip()
        }


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(src, dst)


def split_dataset(
    dataset_dir: str,
    selected_frames_file: str,
    output_ground_truth: str,
    output_predictions: str,
):

    dataset_dir = Path(dataset_dir)

    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"

    gt_dir = Path(output_ground_truth)
    pred_dir = Path(output_predictions)

    selected_frames = load_selected_frames(
        Path(selected_frames_file)
    )

    gt_count = 0
    pred_count = 0

    image_files = sorted(
        images_dir.glob("*.png")
    )

    for image_path in image_files:

        image_name = image_path.name

        label_path = (
            labels_dir
            / image_name.replace(
                ".png",
                ".txt",
            )
        )

        if image_name in selected_frames:

            copy_file(
                image_path,
                gt_dir
                / "images"
                / "train"
                / image_name,
            )

            if label_path.exists():

                copy_file(
                    label_path,
                    gt_dir
                    / "labels"
                    / "train"
                    / label_path.name,
                )

            gt_count += 1

        else:

            copy_file(
                image_path,
                pred_dir
                / "images"
                / "train"
                / image_name,
            )

            pred_count += 1

    print("\n====================")
    print("DONE")
    print("====================")

    print(f"Ground Truth:    {gt_count}")
    print(f"For Prediction:  {pred_count}")
    print(f"Total Images:    {gt_count + pred_count}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
    )

    parser.add_argument(
        "--selected-frames",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--for-predictions",
        required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    split_dataset(
        dataset_dir=args.dataset_dir,
        selected_frames_file=args.selected_frames,
        output_ground_truth=args.ground_truth,
        output_predictions=args.for_predictions,
    )


if __name__ == "__main__":
    main()