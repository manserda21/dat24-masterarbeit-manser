# =========================================================
# detect_possessions.py
# =========================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# ARGUMENTS
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Detect player possessions from tracking CSV"
    )

    parser.add_argument(
        "--tracking_csv",
        type=str,
        required=True,
        help="Path to tracking_results_filtered.csv"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=25,
        help="Maximum BEV distance between player and ball"
    )

    parser.add_argument(
        "--ball_speed_threshold",
        type=float,
        default=35,
        help="Maximum ball speed for controlled possession"
    )

    parser.add_argument(
        "--min_frames",
        type=int,
        default=8,
        help="Minimum possession duration"
    )

    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================

def euclidean_distance(x1, y1, x2, y2):

    return np.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
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

    output_csv = (
        output_dir / "detected_possessions.csv"
    )

    print("\n==============================")
    print("DETECT POSSESSIONS")
    print("==============================")

    print(f"Tracking CSV: {args.tracking_csv}")

    # -----------------------------------------------------
    # LOAD CSV
    # -----------------------------------------------------

    df = pd.read_csv(args.tracking_csv)

    print(f"\nLoaded rows: {len(df)}")

    # -----------------------------------------------------
    # SPLIT PLAYERS / BALL
    # -----------------------------------------------------

    player_df = df[
        df["class_name"].isin(
            ["modric", "kroos"]
        )
    ].copy()

    ball_df = df[
        df["class_name"] == "ball"
    ].copy()
    
    ball_df = (
        ball_df
        .sort_values(
            "confidence",
            ascending=False
        )
        .drop_duplicates(
            subset=["frame"],
        )
        .sort_values("frame")
    )

    # -----------------------------------------------------
    # BALL POSITIONS
    # -----------------------------------------------------

    ball_positions = {}

    previous_ball = None

    for _, row in ball_df.iterrows():

        frame = int(row["frame"])

        bx = row["bev_x"]
        by = row["bev_y"]

        ball_speed = 0

        if previous_ball is not None:

            prev_frame, prev_x, prev_y = previous_ball

            if frame > prev_frame:

                ball_speed = euclidean_distance(
                    bx,
                    by,
                    prev_x,
                    prev_y
                )

        ball_positions[frame] = {
            "x": bx,
            "y": by,
            "speed": ball_speed
        }

        previous_ball = (
            frame,
            bx,
            by
        )

    # -----------------------------------------------------
    # DETECT FRAME-WISE POSSESSION
    # -----------------------------------------------------

    frame_possessions = []

    grouped_frames = player_df.groupby("frame")

    for frame, frame_players in grouped_frames:

        frame = int(frame)

        if frame not in ball_positions:
            continue

        ball_x = ball_positions[frame]["x"]
        ball_y = ball_positions[frame]["y"]
        ball_speed = ball_positions[frame]["speed"]

        if ball_speed > args.ball_speed_threshold:
            continue

        best_player = None
        best_distance = np.inf

        for _, player in frame_players.iterrows():

            px = player["bev_x"]
            py = player["bev_y"]

            distance = euclidean_distance(
                px,
                py,
                ball_x,
                ball_y
            )

            if (
                distance < args.distance_threshold and
                distance < best_distance
            ):

                best_distance = distance

                best_player = {
                    "frame": frame,
                    "player": player["class_name"],
                    "track_key": player["track_key"],
                    "distance": distance,
                    "ball_speed": ball_speed,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "player_x": px,
                    "player_y": py,
                }

        if best_player is not None:
            frame_possessions.append(best_player)

    # -----------------------------------------------------
    # TEMPORAL SMOOTHING
    # -----------------------------------------------------

    possessions = []

    if len(frame_possessions) == 0:

        print("\nNo possessions detected.")
        return

    current = frame_possessions[0]

    start_frame = current["frame"]
    previous_frame = current["frame"]

    distances = [current["distance"]]
    ball_speeds = [current["ball_speed"]]

    for i in range(1, len(frame_possessions)):

        row = frame_possessions[i]

        same_player = (
            row["player"] == current["player"]
        )

        consecutive = (
            row["frame"] <= previous_frame + 3
        )

        if same_player and consecutive:

            previous_frame = row["frame"]

            distances.append(
                row["distance"]
            )

            ball_speeds.append(
                row["ball_speed"]
            )

        else:

            duration = (
                previous_frame - start_frame + 1
            )

            if duration >= args.min_frames:

                possessions.append({
                    "player": current["player"],
                    "track_key": current["track_key"],
                    "start_frame": start_frame,
                    "end_frame": previous_frame,
                    "duration_frames": duration,
                    "avg_distance": np.mean(distances),
                    "avg_ball_speed": np.mean(ball_speeds),
                })

            current = row

            start_frame = row["frame"]
            previous_frame = row["frame"]

            distances = [row["distance"]]
            ball_speeds = [row["ball_speed"]]

    # -----------------------------------------------------
    # FINAL SEGMENT
    # -----------------------------------------------------

    duration = (
        previous_frame - start_frame + 1
    )

    if duration >= args.min_frames:

        possessions.append({
            "player": current["player"],
            "track_key": current["track_key"],
            "start_frame": start_frame,
            "end_frame": previous_frame,
            "duration_frames": duration,
            "avg_distance": np.mean(distances),
            "avg_ball_speed": np.mean(ball_speeds),
        })

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    possession_df = pd.DataFrame(
        possessions
    )

    possession_df.to_csv(
        output_csv,
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(f"\nDetected possessions:")
    print(len(possession_df))

    print(f"\nSaved:")
    print(output_csv)

    print("\nTop possessions:\n")

    print(
        possession_df
        .sort_values(
            "duration_frames",
            ascending=False
        )
        .head(10)
    )

    print()


if __name__ == "__main__":
    main()