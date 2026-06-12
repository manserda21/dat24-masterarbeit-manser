# frame_to_bev.py

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_COLORS = {
    0: (0, 0, 255),        # opponent
    1: (255, 0, 0),        # danilo
    2: (0, 255, 0),        # varane
    3: (255, 255, 0),      # ramos
    4: (255, 0, 255),      # marcelo
    5: (0, 255, 255),      # modric
    6: (128, 0, 255),      # casemiro
    7: (255, 128, 0),      # kroos
    8: (0, 128, 255),      # isco
    9: (128, 255, 0),      # benzema
    10: (255, 0, 128),     # ronaldo
    11: (255, 255, 255),   # ball
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--homographies", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--conf", type=float, default=0.5)

    parser.add_argument("--pitch-width", type=float, default=105.0)
    parser.add_argument("--pitch-height", type=float, default=68.0)
    parser.add_argument("--scale", type=int, default=10)

    parser.add_argument("--ball-class", type=int, default=11)
    parser.add_argument("--ball-conf", type=float, default=0.4)

    return parser.parse_args()


def create_pitch(pitch_width, pitch_height, scale):

    bev_width = int(pitch_width * scale)
    bev_height = int(pitch_height * scale)

    pitch = np.zeros(
        (bev_height, bev_width, 3),
        dtype=np.uint8
    )

    pitch[:] = (40, 120, 40)

    white = (255, 255, 255)

    # =========================
    # Außenlinien
    # =========================

    cv2.rectangle(
        pitch,
        (0, 0),
        (bev_width - 1, bev_height - 1),
        white,
        2
    )

    # =========================
    # Mittellinie
    # =========================

    cv2.line(
        pitch,
        (bev_width // 2, 0),
        (bev_width // 2, bev_height),
        white,
        2
    )

    # =========================
    # Mittelkreis
    # =========================

    center_x = bev_width // 2
    center_y = bev_height // 2

    cv2.circle(
        pitch,
        (center_x, center_y),
        int(9.15 * scale),
        white,
        2
    )

    cv2.circle(
        pitch,
        (center_x, center_y),
        4,
        white,
        -1
    )

    # =========================
    # Strafräume
    # =========================

    penalty_length = int(16.5 * scale)
    penalty_width = int(40.32 * scale)

    penalty_y1 = int(
        (bev_height - penalty_width) / 2
    )

    penalty_y2 = penalty_y1 + penalty_width

    # links

    cv2.rectangle(
        pitch,
        (0, penalty_y1),
        (penalty_length, penalty_y2),
        white,
        2
    )

    # rechts

    cv2.rectangle(
        pitch,
        (bev_width - penalty_length, penalty_y1),
        (bev_width, penalty_y2),
        white,
        2
    )

    # =========================
    # Fünfmeterraum
    # =========================

    goal_area_length = int(5.5 * scale)
    goal_area_width = int(18.32 * scale)

    goal_area_y1 = int(
        (bev_height - goal_area_width) / 2
    )

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

    # =========================
    # Elfmeterpunkte
    # =========================

    left_penalty_spot = (
        int(11.0 * scale),
        center_y
    )

    right_penalty_spot = (
        int((pitch_width - 11.0) * scale),
        center_y
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


def extract_frame_number(name):

    match = re.search(
        r"frame_(\d+)",
        name
    )

    if match is None:
        raise ValueError(
            f"Could not parse frame number from {name}"
        )

    return int(match.group(1))


def load_homography(image_path, npz_path):

    frame_name = Path(image_path).name

    data = np.load(npz_path)

    if frame_name not in data.files:
        raise ValueError(
            f"{frame_name} not found in homography file"
        )

    return data[frame_name]


def project_point(point, H):

    pt = np.array(
        [[[point[0], point[1]]]],
        dtype=np.float32
    )

    bev_pt = cv2.perspectiveTransform(
        pt,
        H
    )

    return (
        float(bev_pt[0][0][0]),
        float(bev_pt[0][0][1])
    )

def get_class_color(cls_id):

    rng = np.random.default_rng(cls_id)

    color = rng.integers(
        low=50,
        high=255,
        size=3
    )

    return (
        int(color[0]),
        int(color[1]),
        int(color[2])
    )

def main():

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(args.image)

    if image is None:
        raise RuntimeError(
            f"Could not load image: {args.image}"
        )

    H = load_homography(
        args.image,
        args.homographies
    )

    model = YOLO(args.model)

    results = model.predict(
        source=image,
        conf=args.conf,
        imgsz=1280,
        verbose=False,
        agnostic_nms=True
    )

    result = results[0]

    class_names = model.names

    detection_image = image.copy()

    pitch = create_pitch(
        args.pitch_width,
        args.pitch_height,
        args.scale
    )

    bev_width = int(
        args.pitch_width * args.scale
    )

    bev_height = int(
        args.pitch_height * args.scale
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

            class_name = class_names[cls_id]
            
            if (
                cls_id == args.ball_class
                and conf < args.ball_conf
            ):
                continue

            image_x = (x1 + x2) / 2
            image_y = y2

            bev_x, bev_y = project_point(
                (image_x, image_y),
                H
            )

            color = CLASS_COLORS.get(
                cls_id,
                (200, 200, 200)
            )

            cv2.rectangle(
                detection_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            cv2.putText(
                detection_image,
                f"{class_name} {conf:.2f}",
                (
                    int(x1),
                    max(20, int(y1) - 5)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            if (
                0 <= bev_x < bev_width
                and
                0 <= bev_y < bev_height
            ):

                cv2.circle(
                    pitch,
                    (
                        int(bev_x),
                        int(bev_y)
                    ),
                    6,
                    color,
                    -1
                )

                cv2.putText(
                    pitch,
                    class_name,
                    (
                        int(bev_x) + 5,
                        int(bev_y)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

    detection_output = (
        output_dir /
        "frame_with_detections.png"
    )

    bev_output = (
        output_dir /
        "frame_bev.png"
    )

    cv2.imwrite(
        str(detection_output),
        detection_image
    )

    cv2.imwrite(
        str(bev_output),
        pitch
    )

    print()
    print("==============================")
    print("DONE")
    print("==============================")
    print(f"Detections: {detection_output}")
    print(f"BEV:        {bev_output}")
    print()


if __name__ == "__main__":
    main()