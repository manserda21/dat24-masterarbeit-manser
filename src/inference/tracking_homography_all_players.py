# tracking_homography_all_players.py

import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track all players and ball, project detections to BEV using homographies."
    )

    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--model", required=True, help="YOLO model path")
    parser.add_argument("--homographies", required=True, help="Homographies .npz path")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument("--pitch-width", type=float, default=105.0)
    parser.add_argument("--pitch-height", type=float, default=68.0)
    parser.add_argument("--scale", type=int, default=10)

    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--ball-conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")

    parser.add_argument("--ball-class", type=int, default=11)
    parser.add_argument("--max-player-trail", type=int, default=200)
    parser.add_argument("--max-ball-trail", type=int, default=80)

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
        raise ValueError(f"No valid homographies found in {path}")

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


def project_point(point, homography):
    pt = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32
    )

    bev_pt = cv2.perspectiveTransform(
        pt,
        homography
    )

    return (
        float(bev_pt[0][0][0]),
        float(bev_pt[0][0][1])
    )


def get_track_color(track_key):
    digest = hashlib.md5(
        str(track_key).encode("utf-8")
    ).digest()

    return (
        60 + digest[0] % 180,
        60 + digest[1] % 180,
        60 + digest[2] % 180,
    )


def create_pitch(pitch_width, pitch_height, scale):
    bev_width = int(pitch_width * scale)
    bev_height = int(pitch_height * scale)

    pitch = np.zeros(
        (bev_height, bev_width, 3),
        dtype=np.uint8
    )

    pitch[:, :] = (40, 120, 40)
    white = (255, 255, 255)

    cv2.rectangle(
        pitch,
        (0, 0),
        (bev_width - 1, bev_height - 1),
        white,
        2
    )

    cv2.line(
        pitch,
        (bev_width // 2, 0),
        (bev_width // 2, bev_height),
        white,
        2
    )

    cv2.circle(
        pitch,
        (bev_width // 2, bev_height // 2),
        int(9.15 * scale),
        white,
        2
    )

    cv2.circle(
        pitch,
        (bev_width // 2, bev_height // 2),
        4,
        white,
        -1
    )

    penalty_length = int(16.5 * scale)
    penalty_width = int(40.32 * scale)
    penalty_y1 = int((pitch_height * scale - penalty_width) / 2)
    penalty_y2 = penalty_y1 + penalty_width

    cv2.rectangle(
        pitch,
        (0, penalty_y1),
        (penalty_length, penalty_y2),
        white,
        2
    )

    cv2.rectangle(
        pitch,
        (bev_width - penalty_length, penalty_y1),
        (bev_width, penalty_y2),
        white,
        2
    )

    goal_area_length = int(5.5 * scale)
    goal_area_width = int(18.32 * scale)
    goal_area_y1 = int((pitch_height * scale - goal_area_width) / 2)
    goal_area_y2 = goal_area_y1 + goal_area_width

    cv2.rectangle(
        pitch,
        (0, goal_area_y1),
        (goal_area_length, goal_area_y2),
        white,
        2
    )

    cv2.rectangle(
        pitch,
        (bev_width - goal_area_length, goal_area_y1),
        (bev_width, goal_area_y2),
        white,
        2
    )

    left_penalty_spot = (
        int(11.0 * scale),
        int(pitch_height * scale / 2)
    )

    right_penalty_spot = (
        int((pitch_width - 11.0) * scale),
        int(pitch_height * scale / 2)
    )

    cv2.circle(
        pitch,
        left_penalty_spot,
        4,
        white,
        -1
    )

    cv2.circle(
        pitch,
        right_penalty_spot,
        4,
        white,
        -1
    )

    return pitch


def is_inside_pitch(bev_x, bev_y, pitch_width, pitch_height, scale):
    return (
        0 <= bev_x < pitch_width * scale
        and
        0 <= bev_y < pitch_height * scale
    )


def draw_trajectory(pitch, points, color, label, pitch_width, pitch_height, scale, thickness=2):
    valid_points = [
        (int(x), int(y))
        for x, y in points
        if is_inside_pitch(
            x,
            y,
            pitch_width,
            pitch_height,
            scale
        )
    ]

    for i in range(1, len(valid_points)):
        cv2.line(
            pitch,
            valid_points[i - 1],
            valid_points[i],
            color,
            thickness
        )

    if valid_points:
        cv2.circle(
            pitch,
            valid_points[-1],
            6,
            color,
            -1
        )

        cv2.putText(
            pitch,
            label,
            (
                valid_points[-1][0] + 6,
                valid_points[-1][1]
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )


def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (
            int(x1),
            max(20, int(y1) - 5)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA
    )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_tracking_video = output_dir / "tracking_video.mp4"
    output_bev_video = output_dir / "bev_tracking.mp4"
    output_csv = output_dir / "bev_positions.csv"
    output_final_image = output_dir / "final_trajectories.png"

    bev_width = int(args.pitch_width * args.scale)
    bev_height = int(args.pitch_height * args.scale)

    print()
    print("==============================")
    print("TRACKING + BEV PROJECTION")
    print("==============================")
    print(f"Video:        {args.video}")
    print(f"Model:        {args.model}")
    print(f"Homographies: {args.homographies}")
    print(f"Output dir:   {output_dir}")
    print(f"Ball class:   {args.ball_class}")
    print(f"Conf:         {args.conf}")
    print(f"Ball conf:    {args.ball_conf}")
    print()

    model = YOLO(args.model)
    class_names = model.names
    
    print("\nDetected classes:")
    for class_id, class_name in class_names.items():
        print(f"{class_id}: {class_name}")

    homographies = load_homographies(
        args.homographies
    )

    print(f"Loaded homographies: {len(homographies)}")
    print(f"Available homography frames: {sorted(homographies.keys())}")
    print()

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    ret, first_frame = cap.read()

    if not ret:
        raise RuntimeError(
            "Could not read first frame."
        )

    frame_height, frame_width = first_frame.shape[:2]

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        0
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tracking_writer = cv2.VideoWriter(
        str(output_tracking_video),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    bev_writer = cv2.VideoWriter(
        str(output_bev_video),
        fourcc,
        fps,
        (bev_width, bev_height)
    )

    print(f"FPS:         {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Frame size:  {frame_width}x{frame_height}")
    print()

    player_trajs = defaultdict(list)
    player_trajs_full = defaultdict(list)

    ball_traj = []
    ball_traj_full = []

    rows = []

    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        H, nearest_h_frame = get_nearest_homography(
            frame_idx,
            homographies
        )

        results = model.track(
            frame_bgr,
            conf=args.conf,
            persist=True,
            verbose=False,
            tracker=args.tracker,
            imgsz=args.imgsz,
            agnostic_nms=True,
        )

        result = results[0]

        pitch = create_pitch(
            args.pitch_width,
            args.pitch_height,
            args.scale
        )

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

                if track_id is None:
                    continue

                class_name = (
                    class_names[cls_id]
                    if cls_id in class_names
                    else str(cls_id)
                )

                image_x = (x1 + x2) / 2
                image_y = y2

                bev_x, bev_y = project_point(
                    (image_x, image_y),
                    H
                )

                if (
                    not np.isfinite(bev_x)
                    or not np.isfinite(bev_y)
                ):
                    continue

                if not is_inside_pitch(
                    bev_x,
                    bev_y,
                    args.pitch_width,
                    args.pitch_height,
                    args.scale
                ):
                    continue

                if cls_id == args.ball_class:
                    if conf < args.ball_conf:
                        continue

                    object_type = "ball"
                    track_key = f"ball_{track_id}"
                    color = (255, 255, 255)

                    ball_traj.append(
                        (bev_x, bev_y)
                    )

                    ball_traj_full.append(
                        (bev_x, bev_y)
                    )

                    if len(ball_traj) > args.max_ball_trail:
                        ball_traj.pop(0)

                    draw_box(
                        frame_bgr,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"ball ID {track_id} {conf:.2f}",
                        color
                    )

                else:
                    object_type = "player"
                    player_key = f"{class_name}_{track_id}"
                    track_key = player_key
                    color = get_track_color(
                        player_key
                    )

                    player_trajs[player_key].append(
                        (bev_x, bev_y)
                    )

                    player_trajs_full[player_key].append(
                        (bev_x, bev_y)
                    )

                    if len(player_trajs[player_key]) > args.max_player_trail:
                        player_trajs[player_key].pop(0)

                    draw_box(
                        frame_bgr,
                        x1,
                        y1,
                        x2,
                        y2,
                        f"{class_name} ID {track_id} {conf:.2f}",
                        color
                    )

                rows.append({
                    "frame": frame_idx,
                    "object_type": object_type,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "track_key": track_key,
                    "confidence": conf,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "image_x": image_x,
                    "image_y": image_y,
                    "bev_x": bev_x,
                    "bev_y": bev_y,
                    "nearest_homography_frame": nearest_h_frame,
                })

        for player_key, traj in player_trajs.items():
            color = get_track_color(
                player_key
            )

            draw_trajectory(
                pitch,
                traj,
                color,
                player_key,
                args.pitch_width,
                args.pitch_height,
                args.scale,
                thickness=2
            )

        draw_trajectory(
            pitch,
            ball_traj,
            (255, 255, 255),
            "ball",
            args.pitch_width,
            args.pitch_height,
            args.scale,
            thickness=1
        )

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

        tracking_writer.write(
            frame_bgr
        )

        bev_writer.write(
            pitch
        )

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(
                f"Processed {frame_idx}/{frame_count} frames"
            )

    cap.release()
    tracking_writer.release()
    bev_writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(
        output_csv,
        index=False
    )

    final_pitch = create_pitch(
        args.pitch_width,
        args.pitch_height,
        args.scale
    )

    for player_key, traj in player_trajs_full.items():
        color = get_track_color(
            player_key
        )

        draw_trajectory(
            final_pitch,
            traj,
            color,
            player_key,
            args.pitch_width,
            args.pitch_height,
            args.scale,
            thickness=2
        )

    draw_trajectory(
        final_pitch,
        ball_traj_full,
        (255, 255, 255),
        "ball",
        args.pitch_width,
        args.pitch_height,
        args.scale,
        thickness=1
    )

    cv2.imwrite(
        str(output_final_image),
        final_pitch
    )

    print()
    print("==============================")
    print("DONE")
    print("==============================")
    print(f"Saved tracking video: {output_tracking_video}")
    print(f"Saved BEV video:      {output_bev_video}")
    print(f"Saved CSV:            {output_csv}")
    print(f"Saved final image:    {output_final_image}")
    print(f"Processed frames:     {frame_idx}")
    print()


if __name__ == "__main__":
    main()