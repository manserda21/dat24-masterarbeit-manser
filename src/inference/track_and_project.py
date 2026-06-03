# track_and_project.py

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 10

CLASS_NAMES = {
    0: "modric",
    1: "ball",
    2: "kroos",
}

CLASS_COLORS = {
    "modric": (0, 0, 255),
    "ball": (255, 255, 255),
    "kroos": (255, 215, 0),
}

PLAYER_CLASSES = {"modric", "kroos"}

MIN_TRACK_LENGTH = 20
MAX_PLAYER_MOVE_DISTANCE = 80
MAX_BALL_MOVE_DISTANCE = 160
BALL_MIN_CONF = 0.55


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Track objects and project detections to bird's-eye-view pitch coordinates"
    )

    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--homography", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)

    parser.add_argument("--min_track_length", type=int, default=MIN_TRACK_LENGTH)
    parser.add_argument("--max_player_move", type=float, default=MAX_PLAYER_MOVE_DISTANCE)
    parser.add_argument("--max_ball_move", type=float, default=MAX_BALL_MOVE_DISTANCE)
    parser.add_argument("--ball_min_conf", type=float, default=BALL_MIN_CONF)

    return parser.parse_args()


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

    if not homographies:
        raise ValueError("No valid homographies found.")

    return homographies


def get_nearest_homography(frame_idx, homographies):
    available_frames = np.array(sorted(homographies.keys()))
    nearest_idx = np.argmin(np.abs(available_frames - frame_idx))
    nearest_frame = int(available_frames[nearest_idx])

    return homographies[nearest_frame], nearest_frame


def project_point(point, homography):
    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    bev_pt = cv2.perspectiveTransform(pt, homography)

    x = float(bev_pt[0][0][0])
    y = float(bev_pt[0][0][1])

    return x, y


def create_pitch():
    bev_width = int(PITCH_WIDTH * SCALE)
    bev_height = int(PITCH_HEIGHT * SCALE)

    pitch = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    pitch[:, :] = (40, 120, 40)

    cv2.rectangle(
        pitch,
        (0, 0),
        (bev_width - 1, bev_height - 1),
        (255, 255, 255),
        2,
    )

    cv2.line(
        pitch,
        (bev_width // 2, 0),
        (bev_width // 2, bev_height),
        (255, 255, 255),
        2,
    )

    cv2.circle(
        pitch,
        (bev_width // 2, bev_height // 2),
        int(9.15 * SCALE),
        (255, 255, 255),
        2,
    )

    return pitch


def is_inside_pitch(bev_x, bev_y):
    bev_width = PITCH_WIDTH * SCALE
    bev_height = PITCH_HEIGHT * SCALE

    return 0 <= bev_x < bev_width and 0 <= bev_y < bev_height


def get_max_move_distance(class_name, args):
    if class_name == "ball":
        return args.max_ball_move

    return args.max_player_move


def is_plausible_movement(track_points, new_point, class_name, args):
    if not track_points:
        return True

    prev_x, prev_y = track_points[-1]
    new_x, new_y = new_point

    distance = np.sqrt((new_x - prev_x) ** 2 + (new_y - prev_y) ** 2)
    max_distance = get_max_move_distance(class_name, args)

    return distance <= max_distance


def draw_trajectory(pitch, points, color, thickness=2):
    valid_points = [
        (int(x), int(y))
        for x, y in points
        if is_inside_pitch(x, y)
    ]

    for i in range(1, len(valid_points)):
        cv2.line(
            pitch,
            valid_points[i - 1],
            valid_points[i],
            color,
            thickness,
        )

    if valid_points:
        cv2.circle(
            pitch,
            valid_points[-1],
            5,
            color,
            -1,
        )


def draw_legend(pitch):

    legend_items = [
        ("modric", CLASS_COLORS["modric"]),
        ("kroos", CLASS_COLORS["kroos"]),
        ("ball", CLASS_COLORS["ball"]),
    ]

    start_x = 20
    start_y = 90

    for i, (label, color) in enumerate(legend_items):

        y = start_y + i * 30

        # color box
        cv2.rectangle(
            pitch,
            (start_x, y - 15),
            (start_x + 20, y + 5),
            color,
            -1
        )

        # label
        cv2.putText(
            pitch,
            label,
            (start_x + 30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

def select_longest_track_per_player_class(trajectories, track_metadata):
    selected_keys = set()

    for class_name in PLAYER_CLASSES:
        candidate_keys = [
            key
            for key, metadata in track_metadata.items()
            if metadata["class_name"] == class_name
        ]

        if not candidate_keys:
            continue

        best_key = max(
            candidate_keys,
            key=lambda key: len(trajectories[key])
        )

        selected_keys.add(best_key)

    return selected_keys


def draw_tracks(
    pitch,
    trajectories,
    track_metadata,
    min_track_length,
    only_selected_keys=None,
):
    for track_key, points in trajectories.items():
        if len(points) < min_track_length:
            continue

        if only_selected_keys is not None and track_key not in only_selected_keys:
            continue

        metadata = track_metadata.get(track_key)

        if metadata is None:
            continue

        class_name = metadata["class_name"]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        thickness = 1 if class_name == "ball" else 2

        draw_trajectory(
            pitch,
            points,
            color=color,
            thickness=thickness,
        )

        last_x, last_y = points[-1]

        cv2.putText(
            pitch,
            track_key,
            (int(last_x) + 5, int(last_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )


def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / "bev_tracking.mp4"
    output_csv_path = output_dir / "tracking_results.csv"
    output_filtered_csv_path = output_dir / "tracking_results_filtered.csv"
    output_image_path = output_dir / "final_trajectories.png"
    output_selected_image_path = output_dir / "final_selected_player_trajectories.png"

    print("\n==============================")
    print("TRACK AND PROJECT")
    print("==============================")
    print(f"Video:       {args.video}")
    print(f"Model:       {args.model}")
    print(f"Homography:  {args.homography}")
    print(f"Output dir:  {output_dir}")
    print(f"Conf:        {args.conf}")
    print(f"Ball conf:   {args.ball_min_conf}")
    print(f"Tracker:     {args.tracker}")
    print(f"Image size:  {args.imgsz}")

    model = YOLO(args.model)
    homographies = load_homographies(args.homography)

    print(f"\nLoaded homographies: {len(homographies)}")

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS:          {fps}")
    print(f"Frame count:  {frame_count}")

    bev_width = int(PITCH_WIDTH * SCALE)
    bev_height = int(PITCH_HEIGHT * SCALE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (bev_width, bev_height),
    )

    rows = []
    filtered_rows = []

    trajectories = defaultdict(list)
    track_metadata = {}

    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        H, nearest_h_frame = get_nearest_homography(frame_idx, homographies)

        results = model.track(
            frame_bgr,
            conf=args.conf,
            persist=True,
            verbose=False,
            tracker=args.tracker,
            imgsz=args.imgsz,
        )

        result = results[0]
        pitch = create_pitch()

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])

                if cls_id not in CLASS_NAMES:
                    continue

                class_name = CLASS_NAMES[cls_id]
                confidence = float(box.conf[0])

                if class_name == "ball" and confidence < args.ball_min_conf:
                    continue

                if box.id is None:
                    continue

                track_id = int(box.id[0])
                track_key = f"{class_name}_{track_id}"

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                image_x = (x1 + x2) / 2
                image_y = y2

                bev_x, bev_y = project_point(
                    (image_x, image_y),
                    H,
                )

                raw_row = {
                    "frame": frame_idx,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "track_key": track_key,
                    "confidence": confidence,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "image_x": image_x,
                    "image_y": image_y,
                    "bev_x": bev_x,
                    "bev_y": bev_y,
                    "nearest_homography_frame": nearest_h_frame,
                }

                rows.append(raw_row)

                if not np.isfinite(bev_x) or not np.isfinite(bev_y):
                    continue

                if not is_inside_pitch(bev_x, bev_y):
                    continue

                new_point = (bev_x, bev_y)

                if not is_plausible_movement(
                    trajectories[track_key],
                    new_point,
                    class_name,
                    args,
                ):
                    continue

                trajectories[track_key].append(new_point)

                track_metadata[track_key] = {
                    "class_name": class_name,
                    "track_id": track_id,
                }

                filtered_rows.append(raw_row)

        draw_tracks(
            pitch=pitch,
            trajectories=trajectories,
            track_metadata=track_metadata,
            min_track_length=args.min_track_length,
        )

        cv2.putText(
            pitch,
            f"Frame: {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            pitch,
            f"Homography frame: {nearest_h_frame}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        
        draw_legend(pitch)

        writer.write(pitch)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count}")

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    df_filtered = pd.DataFrame(filtered_rows)
    df_filtered.to_csv(output_filtered_csv_path, index=False)

    final_pitch = create_pitch()

    draw_tracks(
        pitch=final_pitch,
        trajectories=trajectories,
        track_metadata=track_metadata,
        min_track_length=args.min_track_length,
    )

    cv2.imwrite(str(output_image_path), final_pitch)

    selected_player_keys = select_longest_track_per_player_class(
        trajectories,
        track_metadata,
    )

    selected_pitch = create_pitch()

    draw_tracks(
        pitch=selected_pitch,
        trajectories=trajectories,
        track_metadata=track_metadata,
        min_track_length=args.min_track_length,
        only_selected_keys=selected_player_keys,
    )

    cv2.imwrite(str(output_selected_image_path), selected_pitch)

    print("\n==============================")
    print("DONE")
    print("==============================")
    print(f"Saved video:        {output_video_path}")
    print(f"Saved CSV:          {output_csv_path}")
    print(f"Saved filtered CSV: {output_filtered_csv_path}")
    print(f"Saved image:        {output_image_path}")
    print(f"Saved selected:     {output_selected_image_path}")

    print("\nSelected player tracks:")
    for key in sorted(selected_player_keys):
        print(f"  {key}: {len(trajectories[key])} points")

    print()


if __name__ == "__main__":
    main()