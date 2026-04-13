from ultralytics import YOLO
import torch


def main() -> None:
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="/data/manser/datasets/modric_yolo/data.yaml",

        # Training
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,

        # Output
        project="/data/manser/runs",
        name="modric_v1",

        # Verbesserungen
        patience=30, # Early stopping
        save=True,
        save_period=10,

        # Augmentations (wegen kleinem Dataset)
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,

        # Stabilität
        workers=4,
    )


if __name__ == "__main__":
    main()