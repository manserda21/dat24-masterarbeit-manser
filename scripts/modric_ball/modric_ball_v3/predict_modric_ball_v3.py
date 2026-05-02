from ultralytics import YOLO

model = YOLO("/data/manser/runs/modric_ball_v3/weights/best.pt")

model.predict(
    source="/data/manser/datasets/modric_ball_dataset_v3_processed/images/val",
    save=True,
    conf=0.3,
    project="/data/manser/test_outputs",
    name="val_predictions_v3"
)