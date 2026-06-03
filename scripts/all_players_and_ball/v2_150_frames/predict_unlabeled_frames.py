# =========================================================
# SCRIPT 2
# predict_unlabeled_frames.py
# =========================================================

from ultralytics import YOLO
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = (
    "/data/manser/runs/all_players_and_ball_v2_yolo11s/weights/best.pt"
)

SOURCE_DIR = (
    "/data/manser/datasets/all_players_and_ball/unlabeled_frames/images"
)

OUTPUT_DIR = (
    "/data/manser/datasets/all_players_and_ball/unlabeled_predictions"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# RUN PREDICTIONS
# =========================================================

model.predict(

    source=SOURCE_DIR,

    imgsz=1280,
    conf=0.25,

    save=True,
    save_txt=True,
    save_conf=True,

    project=OUTPUT_DIR,
    name="predict",

    device=0
)


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("PREDICTIONS FINISHED")
print("==============================")

print(f"Model:")
print(MODEL_PATH)

print(f"\nPredictions:")
print(OUTPUT_DIR)

print("\nDone.\n")