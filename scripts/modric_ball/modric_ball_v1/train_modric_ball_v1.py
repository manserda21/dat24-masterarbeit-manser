from ultralytics import YOLO
import torch


def main() -> None:
    data_yaml = "/data/manser/datasets/modric_ball_v1_processed/data.yaml"

    model = YOLO("yolov8s.pt")

    model.train(
        data=data_yaml,

        # wichtig für Ball
        imgsz=1024,          
        batch=8,             # stabil auf GPU
        epochs=120,          # genug lernen, aber nicht overkill

        device=0,
        workers=4,

        patience=30,

        # Generalisierung
        optimizer="auto",

        # Speicherung
        project="/data/manser/runs",
        name="modric_ball_v1",
        exist_ok=False,

        # Augmentations (für small objects optimiert)
        degrees=3.0,
        translate=0.05,
        scale=0.4,
        fliplr=0.5,

        mosaic=1.0,          # sehr wichtig gegen overfitting
        mixup=0.0,

        # wichtig für kleine Objekte, damit sie nicht immer in den letzten Epochen verschwinden
        close_mosaic=10,

        val=True,
        plots=True,

        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()