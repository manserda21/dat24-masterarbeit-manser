from ultralytics import YOLO
import argparse
from config import DATASET_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    model = YOLO("yolov8s.pt")

    model.train(
        data=args.data,

        imgsz=1024,
        batch=12,
        epochs=120,

        device=0,
        workers=4,
        patience=30,

        optimizer="auto",

        project=RUNS_DIR,
        name=args.name,

        degrees=3.0,
        translate=0.05,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,

        val=True,
        plots=True,

        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()