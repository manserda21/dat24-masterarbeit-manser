from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd

# =========================
# CONFIG
# =========================
MODEL_PATH = "/data/manser/runs/modric_ball_v12/weights/best.pt"
VIDEO_PATH = "/data/manser/test_videos/malaga_modric_clip_10m00s_30s.mp4"

OUTPUT_DIR = Path("/data/manser/test_outputs/modric_ball_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "predictions.mp4"
OUTPUT_CSV_PATH = OUTPUT_DIR / "predictions.csv"

CONF_THRESHOLD = 0.3  # bewusst niedrig zum Debuggen


# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)


# =========================
# VIDEO SETUP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))


# =========================
# STORAGE
# =========================
rows = []

frame_idx = 0


# =========================
# LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    r = results[0]

    # Annotiertes Bild erzeugen
    annotated_frame = r.plot()

    out.write(annotated_frame)

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append({
                "frame_idx": frame_idx,
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            })

    frame_idx += 1


# =========================
# SAVE RESULTS
# =========================
cap.release()
out.release()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Video gespeichert: {OUTPUT_VIDEO_PATH}")
print(f"CSV gespeichert: {OUTPUT_CSV_PATH}")
print(f"Anzahl Predictions: {len(df)}")