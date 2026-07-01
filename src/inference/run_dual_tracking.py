import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import RUNS_DIR
from src.tracking.parquet_exporter import export_rows_to_parquet


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return float(fps)


def create_common_row(
    frame_idx: int,
    fps: float,
    object_type: str,
    team: str,
    track_id,
    class_id,
    class_name: str,
    confidence: float,
    xyxy,
    source_model: str,
) -> dict:
    x_min, y_min, x_max, y_max = xyxy

    width = x_max - x_min
    height = y_max - y_min

    x_center = x_min + width / 2
    y_center = y_min + height / 2

    return {
        "frame": frame_idx,
        "time_seconds": frame_idx / fps,
        "time_ms": int((frame_idx / fps) * 1000),
        "fps": fps,
        "object_type": object_type,
        "team": team,
        "track_id": track_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "x_center": float(x_center),
        "y_center": float(y_center),
        "width": float(width),
        "height": float(height),
        "x_min": float(x_min),
        "y_min": float(y_min),
        "x_max": float(x_max),
        "y_max": float(y_max),
        "source_model": source_model,
    }


def run_dual_tracking(
    player_model_path: str,
    ball_model_path: str,
    source: str,
    fps: float,
    player_conf: float = 0.25,
    ball_conf: float = 0.35,
    iou: float = 0.5,
    player_tracker: str = "botsort.yaml",
    imgsz: int = 1280,
) -> list[dict]:
    player_model = YOLO(player_model_path)
    ball_model = YOLO(ball_model_path)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    player_stream = player_model.track(
        source=source,
        conf=player_conf,
        iou=iou,
        tracker=player_tracker,
        imgsz=imgsz,
        stream=True,
        persist=True,
        verbose=False,
    )

    rows = []

    print("Processing player tracks and ball detections...")

    for frame_idx, player_result in enumerate(player_stream):
        success, frame = cap.read()

        if not success:
            break

        if player_result.boxes is not None:
            boxes = player_result.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                class_name = player_result.names[cls_id]

                team = (
                    "opponent"
                    if class_name == "opponent"
                    else "real_madrid"
                )

                track_id = None

                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i].item())

                xyxy = boxes.xyxy[i].cpu().numpy()

                rows.append(
                    create_common_row(
                        frame_idx=frame_idx,
                        fps=fps,
                        object_type="player",
                        team=team,
                        track_id=track_id,
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=float(boxes.conf[i].item()),
                        xyxy=xyxy,
                        source_model="player_model",
                    )
                )

        ball_results = ball_model.predict(
            source=frame,
            conf=ball_conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )

        ball_result = ball_results[0]

        if ball_result.boxes is not None and len(ball_result.boxes) > 0:
            b_boxes = ball_result.boxes
            b_confs = b_boxes.conf.cpu().numpy()

            best_idx = int(b_confs.argmax())

            xyxy = b_boxes.xyxy[best_idx].cpu().numpy()
            confidence = float(b_boxes.conf[best_idx].item())

            cls_id = int(b_boxes.cls[best_idx].item())
            class_name = ball_result.names.get(cls_id, "ball")

            rows.append(
                create_common_row(
                    frame_idx=frame_idx,
                    fps=fps,
                    object_type="ball",
                    team="ball",
                    track_id=0,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=xyxy,
                    source_model="ball_model",
                )
            )

        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}")

    cap.release()

    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run player tracking and ball detection on a video."
    )

    parser.add_argument(
        "--player_model",
        type=str,
        required=True,
        help="Path to player YOLO model",
    )

    parser.add_argument(
        "--ball_model",
        type=str,
        required=True,
        help="Path to ball YOLO model",
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Output folder name",
    )

    parser.add_argument(
        "--project",
        type=str,
        default=RUNS_DIR,
    )

    parser.add_argument(
        "--player_conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--ball_conf",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--player_tracker",
        type=str,
        default="botsort.yaml",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=== Dual Tracking Start ===")
    print(f"Player Model:   {args.player_model}")
    print(f"Ball Model:     {args.ball_model}")
    print(f"Source:         {args.source}")
    print(f"Player Tracker: {args.player_tracker}")
    print(f"Player Conf:    {args.player_conf}")
    print(f"Ball Conf:      {args.ball_conf}")

    fps = get_video_fps(args.source)

    print(f"FPS: {fps}")

    rows = run_dual_tracking(
        player_model_path=args.player_model,
        ball_model_path=args.ball_model,
        source=args.source,
        fps=fps,
        player_conf=args.player_conf,
        ball_conf=args.ball_conf,
        iou=args.iou,
        player_tracker=args.player_tracker,
        imgsz=args.imgsz,
    )

    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "tracking.parquet"

    export_rows_to_parquet(
        rows=rows,
        output_path=str(parquet_path),
    )

    print("=== Dual Tracking Done ===")


if __name__ == "__main__":
    main()