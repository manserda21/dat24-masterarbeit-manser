# =========================================================
# export_possession_frames.py
# =========================================================

import argparse
from pathlib import Path

import cv2
import pandas as pd


# =========================================================
# COLORS
# =========================================================

CLASS_COLORS = {
    "modric": (0, 0, 255),
    "kroos": (255, 215, 0),
    "ball": (0, 255, 255),
}


# =========================================================
# ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Export possession validation frames"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Original input video"
    )

    parser.add_argument(
        "--possessions_csv",
        type=str,
        required=True,
        help="detected_possessions.csv"
    )

    parser.add_argument(
        "--tracking_csv",
        type=str,
        required=True,
        help="tracking_results_filtered.csv"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Export top N possessions"
    )

    return parser.parse_args()


# =========================================================
# DRAW DETECTIONS
# =========================================================

def draw_tracking_detections(
    frame,
    frame_tracking,
):

    for _, det in frame_tracking.iterrows():

        class_name = det["class_name"]

        x1 = int(det["bbox_x1"])
        y1 = int(det["bbox_y1"])
        x2 = int(det["bbox_x2"])
        y2 = int(det["bbox_y2"])

        color = CLASS_COLORS.get(
            class_name,
            (255, 255, 255)
        )

        thickness = 2

        if class_name == "ball":
            thickness = 3

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        label = (
            f"{class_name}_"
            f"{det['track_id']}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


# =========================================================
# DRAW POSSESSION INFO
# =========================================================

def draw_possession_info(
    frame,
    row,
):

    player = row["player"]

    start_frame = int(
        row["start_frame"]
    )

    end_frame = int(
        row["end_frame"]
    )

    duration = int(
        row["duration_frames"]
    )

    avg_distance = float(
        row["avg_distance"]
    )

    avg_ball_speed = float(
        row["avg_ball_speed"]
    )

    info_lines = [
        f"Player: {player}",
        f"Frames: {start_frame}-{end_frame}",
        f"Duration: {duration}",
        f"Avg distance: {avg_distance:.2f}",
        f"Avg ball speed: {avg_ball_speed:.2f}",
    ]

    start_x = 30
    start_y = 40

    for i, text in enumerate(info_lines):

        y = start_y + i * 35

        cv2.putText(
            frame,
            text,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )


# =========================================================
# DRAW LEGEND
# =========================================================

def draw_legend(frame):

    legend_items = [
        ("modric", CLASS_COLORS["modric"]),
        ("kroos", CLASS_COLORS["kroos"]),
        ("ball", CLASS_COLORS["ball"]),
    ]

    start_x = 30
    start_y = 250

    for i, (label, color) in enumerate(legend_items):

        y = start_y + i * 35

        cv2.rectangle(
            frame,
            (start_x, y - 15),
            (start_x + 20, y + 5),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (start_x + 35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_arguments()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_dir = (
        output_dir / "possession_frames"
    )

    frames_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n==============================")
    print("EXPORT POSSESSION FRAMES")
    print("==============================")

    # -----------------------------------------------------
    # LOAD CSV FILES
    # -----------------------------------------------------

    possession_df = pd.read_csv(
        args.possessions_csv
    )

    tracking_df = pd.read_csv(
        args.tracking_csv
    )

    if len(possession_df) == 0:

        print("\nNo possessions found.")
        return

    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    possession_df = possession_df.sort_values(
        "duration_frames",
        ascending=False
    ).head(args.top_n)

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        args.video
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    # -----------------------------------------------------
    # EXPORT FRAMES
    # -----------------------------------------------------

    for idx, row in possession_df.iterrows():

        player = row["player"]

        start_frame = int(
            row["start_frame"]
        )

        end_frame = int(
            row["end_frame"]
        )

        middle_frame = int(
            (start_frame + end_frame) / 2
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            middle_frame
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # -------------------------------------------------
        # GET TRACKING FOR FRAME
        # -------------------------------------------------

        frame_tracking = tracking_df[
            tracking_df["frame"] == middle_frame
        ]

        # -------------------------------------------------
        # DRAW DETECTIONS
        # -------------------------------------------------

        draw_tracking_detections(
            frame,
            frame_tracking
        )

        # -------------------------------------------------
        # DRAW INFO
        # -------------------------------------------------

        draw_possession_info(
            frame,
            row
        )

        # -------------------------------------------------
        # DRAW LEGEND
        # -------------------------------------------------

        draw_legend(frame)

        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        filename = (
            f"{player}"
            f"_start{start_frame}"
            f"_end{end_frame}"
            f"_mid{middle_frame}.png"
        )

        output_path = (
            frames_dir / filename
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        print(
            f"Saved: {output_path.name}"
        )

    cap.release()

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(f"\nFrames saved to:")
    print(frames_dir)

    print()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()