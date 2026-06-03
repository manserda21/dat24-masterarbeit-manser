# inference_video.py

from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse


# =========================================================
# ARGUMENT PARSER
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run YOLO inference on a video"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained YOLO model"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference image size"
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_arguments()

    print("\n==============================")
    print("YOLO VIDEO INFERENCE")
    print("==============================")

    print(f"Model:  {args.model}")
    print(f"Video:  {args.video}")
    print(f"Output: {args.output}")

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = YOLO(args.model)

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # -----------------------------------------------------
    # OUTPUT VIDEO
    # -----------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -------------------------------------------------
        # INFERENCE
        # -------------------------------------------------

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            agnostic_nms=True,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # -------------------------------------------------
        # WRITE FRAME
        # -------------------------------------------------

        writer.write(annotated_frame)

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")

    # -----------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------

    cap.release()
    writer.release()

    print("\n==============================")
    print("INFERENCE FINISHED")
    print("==============================")

    print(f"\nSaved to:")
    print(output_path)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()