# extract_frames_every_n.py

import cv2
import argparse
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 50,
    image_extension: str = ".png",
):
    """
    Extract every nth frame from a video.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_seconds = total_frames / fps

    print("\n==============================")
    print("VIDEO INFO")
    print("==============================")

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Duration (s): {duration_seconds:.2f}")

    print("\n==============================")
    print("EXTRACTING FRAMES")
    print("==============================\n")

    frame_index = 0
    saved_frames = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_interval == 0:

            output_name = (
                f"frame_{frame_index:06d}"
                f"{image_extension}"
            )

            output_path = (
                output_dir / output_name
            )

            cv2.imwrite(
                str(output_path),
                frame,
            )

            saved_frames += 1

            print(
                f"[{saved_frames}] Saved: "
                f"{output_name}"
            )

        frame_index += 1

    cap.release()

    print("\n==============================")
    print("DONE")
    print("==============================")

    print(f"\nSaved frames: {saved_frames}")

    print("\nOutput directory:")
    print(output_dir)

    print()


def parse_args():

    parser = argparse.ArgumentParser(
        description="Extract every nth frame from a video."
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Save every nth frame",
    )

    parser.add_argument(
        "--ext",
        type=str,
        default=".png",
        help="Output image extension",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        image_extension=args.ext,
    )


if __name__ == "__main__":
    main()