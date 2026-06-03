# tracking_homography.py

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
import re


# =========================
# CONFIG
# =========================

VIDEO_PATH = "/data/manser/test_videos/malaga_modric_clip_10m00s_5min.mp4"

MODEL_PATH = "/data/manser/runs/modric_ball_v3/weights/best.pt"

HOMOGRAPHY_PATH = "/data/manser/homography_outputs/malaga_real_calib_v2/homographies.npz"

OUTPUT_DIR = Path("/data/manser/tracking_outputs/modric_ball_bev_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO_PATH = OUTPUT_DIR / "modric_ball_bev_tracking.mp4"
OUTPUT_CSV_PATH = OUTPUT_DIR / "modric_ball_bev_positions.csv"

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 10

CONF = 0.4

CLASS_MODRIC = 0
CLASS_BALL = 1


# =========================
# HELPER
# =========================

def extract_frame_number(frame_name):

    match = re.search(r"frame_(\d+)", frame_name)

    if match is None:
        return None

    return int(match.group(1))


def load_homographies(path):

    data = np.load(path)

    homographies = {}

    for name in data.files:

        frame_number = extract_frame_number(name)

        if frame_number is not None:
            homographies[frame_number] = data[name]

    return homographies


def get_nearest_homography(frame_idx, homographies):

    available_frames = np.array(
        sorted(homographies.keys())
    )

    nearest_idx = np.argmin(
        np.abs(available_frames - frame_idx)
    )

    nearest_frame = int(
        available_frames[nearest_idx]
    )

    return homographies[nearest_frame], nearest_frame


def project_point(point, H):

    pt = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32
    )

    bev_pt = cv2.perspectiveTransform(pt, H)

    x = float(bev_pt[0][0][0])
    y = float(bev_pt[0][0][1])

    return x, y


def create_pitch():

    bev_width = int(PITCH_WIDTH * SCALE)
    bev_height = int(PITCH_HEIGHT * SCALE)

    pitch = np.zeros(
        (bev_height, bev_width, 3),
        dtype=np.uint8
    )

    # green field
    pitch[:, :] = (40, 120, 40)

    # outer lines
    cv2.rectangle(
        pitch,
        (0, 0),
        (bev_width - 1, bev_height - 1),
        (255, 255, 255),
        2
    )

    # middle line
    cv2.line(
        pitch,
        (bev_width // 2, 0),
        (bev_width // 2, bev_height),
        (255, 255, 255),
        2
    )

    # center circle
    cv2.circle(
        pitch,
        (bev_width // 2, bev_height // 2),
        int(9.15 * SCALE),
        (255, 255, 255),
        2
    )

    return pitch


def draw_trajectory(pitch, points, color, label):

    valid_points = [
        (int(x), int(y))
        for x, y in points
        if (
            0 <= x < PITCH_WIDTH * SCALE and
            0 <= y < PITCH_HEIGHT * SCALE
        )
    ]

    for i in range(1, len(valid_points)):

        thickness = 2

        if label == "ball":
            thickness = 1

        cv2.line(
            pitch,
            valid_points[i - 1],
            valid_points[i],
            color,
            thickness
        )

    if len(valid_points) > 0:

        cv2.circle(
            pitch,
            valid_points[-1],
            7,
            color,
            -1
        )

        cv2.putText(
            pitch,
            label,
            (
                valid_points[-1][0] + 8,
                valid_points[-1][1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )


# =========================
# LOAD
# =========================

model = YOLO(MODEL_PATH)

homographies = load_homographies(HOMOGRAPHY_PATH)

print("Loaded homographies:", len(homographies))
print("Available homography frames:")
print(sorted(homographies.keys()))


# =========================
# VIDEO SETUP
# =========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

bev_width = int(PITCH_WIDTH * SCALE)
bev_height = int(PITCH_HEIGHT * SCALE)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO_PATH),
    fourcc,
    fps,
    (bev_width, bev_height)
)

print("FPS:", fps)
print("Frame count:", frame_count)


# =========================
# TRACKING LOOP
# =========================

# short trajectories for video
modric_traj = []
ball_traj = []

# full trajectories for final image
modric_traj_full = []
ball_traj_full = []

rows = []

frame_idx = 0

while True:

    ret, frame_bgr = cap.read()

    if not ret:
        break

    # =====================================
    # GET NEAREST HOMOGRAPHY
    # =====================================

    H, nearest_h_frame = get_nearest_homography(
        frame_idx,
        homographies
    )

    # =====================================
    # YOLO TRACKING
    # =====================================

    results = model.track(
        frame_bgr,
        conf=CONF,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml"
    )

    result = results[0]

    # =====================================
    # PROCESS DETECTIONS
    # =====================================

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            # =====================================
            # MODRIC
            # =====================================

            if cls_id == CLASS_MODRIC:

                img_x = (x1 + x2) / 2
                img_y = y2

                bev_x, bev_y = project_point(
                    (img_x, img_y),
                    H
                )

                # invalid projection
                if (
                    not np.isfinite(bev_x) or
                    not np.isfinite(bev_y)
                ):
                    continue

                # outside pitch
                if not (
                    0 <= bev_x < bev_width and
                    0 <= bev_y < bev_height
                ):
                    continue

                # short trajectory
                modric_traj.append(
                    (bev_x, bev_y)
                )

                # full trajectory
                modric_traj_full.append(
                    (bev_x, bev_y)
                )

                # keep video trajectory short
                if len(modric_traj) > 200:
                    modric_traj.pop(0)

                rows.append({
                    "frame": frame_idx,
                    "object": "modric",
                    "track_id": track_id,
                    "confidence": conf,
                    "image_x": img_x,
                    "image_y": img_y,
                    "bev_x": bev_x,
                    "bev_y": bev_y,
                    "nearest_homography_frame": nearest_h_frame
                })

            # =====================================
            # BALL
            # =====================================

            elif (
                cls_id == CLASS_BALL and
                conf > 0.5
            ):

                img_x = (x1 + x2) / 2

                # lower edge works often better
                img_y = y2

                bev_x, bev_y = project_point(
                    (img_x, img_y),
                    H
                )

                # invalid projection
                if (
                    not np.isfinite(bev_x) or
                    not np.isfinite(bev_y)
                ):
                    continue

                # outside pitch
                if not (
                    0 <= bev_x < bev_width and
                    0 <= bev_y < bev_height
                ):
                    continue

                # short trajectory
                ball_traj.append(
                    (bev_x, bev_y)
                )

                # full trajectory
                ball_traj_full.append(
                    (bev_x, bev_y)
                )

                # keep video trajectory short
                if len(ball_traj) > 80:
                    ball_traj.pop(0)

                rows.append({
                    "frame": frame_idx,
                    "object": "ball",
                    "track_id": track_id,
                    "confidence": conf,
                    "image_x": img_x,
                    "image_y": img_y,
                    "bev_x": bev_x,
                    "bev_y": bev_y,
                    "nearest_homography_frame": nearest_h_frame
                })

    # =====================================
    # DRAW PITCH
    # =====================================

    pitch = create_pitch()

    draw_trajectory(
        pitch,
        modric_traj,
        color=(0, 0, 255),
        label="modric"
    )

    draw_trajectory(
        pitch,
        ball_traj,
        color=(255, 255, 255),
        label="ball"
    )

    # =====================================
    # INFO TEXT
    # =====================================

    cv2.putText(
        pitch,
        f"Frame: {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        pitch,
        f"Homography frame: {nearest_h_frame}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # =====================================
    # WRITE VIDEO
    # =====================================

    writer.write(pitch)

    frame_idx += 1

    if frame_idx % 100 == 0:

        print(
            f"Processed {frame_idx}/{frame_count} frames"
        )


# =========================
# CLEANUP
# =========================

cap.release()
writer.release()


# =====================================
# SAVE CSV
# =====================================

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_CSV_PATH,
    index=False
)

print("\nDone.")
print("Saved video:", OUTPUT_VIDEO_PATH)
print("Saved CSV:", OUTPUT_CSV_PATH)


# =========================
# FINAL TRAJECTORY IMAGE
# =========================

final_pitch = create_pitch()

# full Modric trajectory
draw_trajectory(
    final_pitch,
    modric_traj_full,
    color=(0, 0, 255),
    label="modric"
)

# full ball trajectory
draw_trajectory(
    final_pitch,
    ball_traj_full,
    color=(255, 255, 255),
    label="ball"
)

FINAL_IMAGE_PATH = (
    OUTPUT_DIR / "final_trajectories.png"
)

cv2.imwrite(
    str(FINAL_IMAGE_PATH),
    final_pitch
)

print(
    "Saved final trajectory image:",
    FINAL_IMAGE_PATH
)