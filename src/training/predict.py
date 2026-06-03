# src/predict_dataset.py

import argparse
from pathlib import Path

from ultralytics import YOLO


def run_predictions(
    model_path: str,
    source_dir: str,
    output_dir: str,
    conf: float = 0.4,
):
    """
    Run YOLO predictions on a directory of images and
    save labels for CVAT correction.
    """

    model_path = Path(model_path)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}"
        )

    print("\n==============================")
    print("YOLO PREDICTION")
    print("==============================")

    print(f"Model: {model_path}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence: {conf}")

    model = YOLO(str(model_path))

    model.predict(
        source=str(source_dir),
        conf=conf,
        save=False,
        save_txt=True,
        save_conf=False,
        agnostic_nms=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        verbose=True,
    )

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(
        "\nLabels saved to:\n"
        f"{output_dir}/predictions/labels"
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate YOLO predictions for CVAT."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model (.pt)",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory with images",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    run_predictions(
        model_path=args.model,
        source_dir=args.source,
        output_dir=args.output,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()