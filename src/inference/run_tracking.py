# run_tracking.py

from ultralytics import YOLO
import argparse
from pathlib import Path
from config import RUNS_DIR


def run_tracking(
    model_path: str,
    source: str,
    output_name: str,
    project: str,
    conf: float = 0.25,
    iou: float = 0.5,
    tracker: str = "bytetrack.yaml",
    save: bool = True,
):
    """
    Führt Tracking auf einem Video aus.
    """

    model = YOLO(model_path)

    results = model.track(
        source=source,
        conf=conf,
        iou=iou,
        tracker=tracker,
        imgsz=1280,
        save=save,
        save_txt=True,
        save_conf=True,
        project=project,
        name=output_name,
        exist_ok=True,
        persist=True,
        verbose=True,
    )

    return results


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="Pfad zu .pt Modell")
    parser.add_argument("--source", type=str, required=True, help="Video Pfad")
    parser.add_argument("--name", type=str, required=True, help="Output Ordner Name")
    parser.add_argument("--project", type=str, default=RUNS_DIR)
    

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Tracking Start ===")
    print(f"Model:  {args.model}")
    print(f"Source: {args.source}")
    print(f"Output: {args.name}")
    print(f"Project: {args.project}")

    run_tracking(
        model_path=args.model,
        source=args.source,
        output_name=args.name,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        project=args.project,
    )

    print("=== Tracking Done ===")


if __name__ == "__main__":
    main()