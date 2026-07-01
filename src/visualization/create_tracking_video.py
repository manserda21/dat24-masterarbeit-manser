import argparse
from pathlib import Path

import cv2
import pandas as pd


CLASS_COLORS = {
    "opponent": (0, 0, 255),
    "ball": (0, 255, 255),
    "benzema": (0, 255, 0),
    "ronaldo": (255, 0, 0),
    "kroos": (255, 128, 0),
    "modric": (255, 0, 255),
    "isco": (0, 255, 128),
    "marcelo": (128, 255, 0),
    "casemiro": (128, 0, 255),
    "ramos": (255, 255, 0),
    "varane": (255, 128, 128),
    "danilo": (128, 255, 255),
}


def get_color(class_name: str):
    return CLASS_COLORS.get(
        class_name,
        (255, 255, 255),
    )


def draw_detection(
    frame,
    row,
    class_column: str,
):
    x1 = int(row["x_min"])
    y1 = int(row["y_min"])
    x2 = int(row["x_max"])
    y2 = int(row["y_max"])

    class_name = str(
        row[class_column])
    track_id = row["track_id"]

    color = get_color(class_name)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    if class_name == "ball":

        label = (
            f"BALL "
            f"{row['confidence']:.2f}"
        )

    else:

        if pd.isna(track_id):
            label = class_name
        else:
            label = (
                f"{class_name} "
                f"({int(track_id)})"
            )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def create_tracking_video(
    video_path: str,
    parquet_path: str,
    output_path: str,
    class_column: str,
):
    print("Loading parquet...")

    df = pd.read_parquet(
        parquet_path
    )

    grouped = {
        frame: frame_df
        for frame, frame_df
        in df.groupby("frame")
    }

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height),
    )

    frame_idx = 0

    print("Rendering video...")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_df = grouped.get(
            frame_idx
        )

        if frame_df is not None:

            for _, row in frame_df.iterrows():

                draw_detection(
                    frame,
                    row,
                    class_column,
                )

        writer.write(
            frame
        )

        if frame_idx % 100 == 0:
            print(
                f"Frame {frame_idx}"
            )

        frame_idx += 1

    cap.release()
    writer.release()

    print(
        f"\nVideo saved to:\n{output_path}"
    )


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Original video",
    )

    parser.add_argument(
        "--parquet",
        required=True,
        help="Tracking parquet",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output video",
    )
    
    parser.add_argument(
        "--class_column",
        default="class_name",
        help="Column used for visualization (default: class_name)",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    create_tracking_video(
        video_path=args.video,
        parquet_path=args.parquet,
        output_path=args.output,
        class_column=args.class_column,
    )


if __name__ == "__main__":
    main()