from ultralytics import YOLO
import torch


def main() -> None:
    data_yaml = "/data/manser/datasets/modric_v2/processed/modric_dataset_v2/data.yaml"

    # Modell auswaehlen: yolov8n.pt oder yolo26n.pt
    # model_path = "yolov8n.pt"
    model_path = "yolo26n.pt"

    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    model = YOLO(model_path)

    model.train(
        data=data_yaml,
        epochs=180,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=40,
        pretrained=True,
        optimizer="auto",

        # Speicherung
        project="/data/manser/runs",
        name="modric_v2_yolo26n",
        exist_ok=False,
        save=True,
        save_period=10,

        # Augmentierung
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,

        # Validierung
        val=True,
        plots=True,

        # Reproduzierbarkeit
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()