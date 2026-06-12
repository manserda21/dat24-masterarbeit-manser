import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
SCALE = 10.0

LINE_CLASSES = [
    "Side line top",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Middle line",
    "Big rect. left top",
    "Big rect. left bottom",
    "Big rect. left main",
    "Small rect. left top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Big rect. right top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Small rect. right top",
    "Small rect. right bottom",
    "Small rect. right main",
]

PITCH_LINES = {
    "Side line top": ((0.0, 0.0), (PITCH_WIDTH, 0.0)),
    "Side line bottom": ((0.0, PITCH_HEIGHT), (PITCH_WIDTH, PITCH_HEIGHT)),
    "Side line left": ((0.0, 0.0), (0.0, PITCH_HEIGHT)),
    "Side line right": ((PITCH_WIDTH, 0.0), (PITCH_WIDTH, PITCH_HEIGHT)),
    "Middle line": ((PITCH_WIDTH / 2.0, 0.0), (PITCH_WIDTH / 2.0, PITCH_HEIGHT)),

    "Big rect. left top": ((0.0, 13.84), (16.5, 13.84)),
    "Big rect. left bottom": ((0.0, 54.16), (16.5, 54.16)),
    "Big rect. left main": ((16.5, 13.84), (16.5, 54.16)),

    "Small rect. left top": ((0.0, 24.84), (5.5, 24.84)),
    "Small rect. left bottom": ((0.0, 43.16), (5.5, 43.16)),
    "Small rect. left main": ((5.5, 24.84), (5.5, 43.16)),

    "Big rect. right top": ((PITCH_WIDTH - 16.5, 13.84), (PITCH_WIDTH, 13.84)),
    "Big rect. right bottom": ((PITCH_WIDTH - 16.5, 54.16), (PITCH_WIDTH, 54.16)),
    "Big rect. right main": ((PITCH_WIDTH - 16.5, 13.84), (PITCH_WIDTH - 16.5, 54.16)),

    "Small rect. right top": ((PITCH_WIDTH - 5.5, 24.84), (PITCH_WIDTH, 24.84)),
    "Small rect. right bottom": ((PITCH_WIDTH - 5.5, 43.16), (PITCH_WIDTH, 43.16)),
    "Small rect. right main": ((PITCH_WIDTH - 5.5, 24.84), (PITCH_WIDTH - 5.5, 43.16)),
}


def homogeneous_line_from_points(p1, p2):
    p1_h = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    p2_h = np.array([p2[0], p2[1], 1.0], dtype=np.float64)

    line = np.cross(p1_h, p2_h)
    norm = np.linalg.norm(line[:2])

    if norm > 0:
        line = line / norm

    return line


def intersect_lines(line_a, line_b):
    point = np.cross(line_a, line_b)

    if abs(point[2]) < 1e-9:
        return None

    return np.array(
        [point[0] / point[2], point[1] / point[2]],
        dtype=np.float64,
    )


def fit_line_from_mask(mask, min_pixels=30):
    ys, xs = np.where(mask > 0.5)

    if len(xs) < min_pixels:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    )

    vx = float(vx[0])
    vy = float(vy[0])
    x0 = float(x0[0])
    y0 = float(y0[0])

    p1 = (x0 - 2000.0 * vx, y0 - 2000.0 * vy)
    p2 = (x0 + 2000.0 * vx, y0 + 2000.0 * vy)

    return homogeneous_line_from_points(p1, p2)


def point_inside_image(point, width, height, margin=150):

    x, y = point

    return (
        -margin <= x <= width + margin
        and -margin <= y <= height + margin
    )


def point_inside_pitch(point):
    x, y = point

    return (
        0.0 <= x <= PITCH_WIDTH
        and 0.0 <= y <= PITCH_HEIGHT
    )


def predict_lines(model, image_path, conf):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    result = model.predict(
        source=str(image_path),
        conf=conf,
        verbose=False,
    )[0]

    predicted_lines = {}

    if result.masks is None:
        return image, predicted_lines

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    for mask, cls_id, score in zip(masks, cls_ids, confs):
        if cls_id < 0 or cls_id >= len(LINE_CLASSES):
            continue

        label = LINE_CLASSES[cls_id]

        mask_resized = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        line = fit_line_from_mask(mask_resized)

        if line is None:
            continue

        if label not in predicted_lines or score > predicted_lines[label]["score"]:
            predicted_lines[label] = {
                "line": line,
                "score": float(score),
            }

    return image, predicted_lines

VALID_INTERSECTIONS = {
    "TL_PITCH_CORNER": (
        "Side line left",
        "Side line top",
    ),
    "TR_PITCH_CORNER": (
        "Side line right",
        "Side line top",
    ),
    "BL_PITCH_CORNER": (
        "Side line left",
        "Side line bottom",
    ),
    "BR_PITCH_CORNER": (
        "Side line right",
        "Side line bottom",
    ),

    "LEFT_BIG_RECT_TOP_LEFT": (
        "Side line left",
        "Big rect. left top",
    ),
    "LEFT_BIG_RECT_BOTTOM_LEFT": (
        "Side line left",
        "Big rect. left bottom",
    ),
    "LEFT_BIG_RECT_TOP_RIGHT": (
        "Big rect. left top",
        "Big rect. left main",
    ),
    "LEFT_BIG_RECT_BOTTOM_RIGHT": (
        "Big rect. left bottom",
        "Big rect. left main",
    ),

    "RIGHT_BIG_RECT_TOP_LEFT": (
        "Big rect. right top",
        "Big rect. right main",
    ),
    "RIGHT_BIG_RECT_BOTTOM_LEFT": (
        "Big rect. right bottom",
        "Big rect. right main",
    ),
    "RIGHT_BIG_RECT_TOP_RIGHT": (
        "Side line right",
        "Big rect. right top",
    ),
    "RIGHT_BIG_RECT_BOTTOM_RIGHT": (
        "Side line right",
        "Big rect. right bottom",
    ),

    "LEFT_SMALL_RECT_TOP_LEFT": (
        "Side line left",
        "Small rect. left top",
    ),
    "LEFT_SMALL_RECT_BOTTOM_LEFT": (
        "Side line left",
        "Small rect. left bottom",
    ),
    "LEFT_SMALL_RECT_TOP_RIGHT": (
        "Small rect. left top",
        "Small rect. left main",
    ),
    "LEFT_SMALL_RECT_BOTTOM_RIGHT": (
        "Small rect. left bottom",
        "Small rect. left main",
    ),

    "RIGHT_SMALL_RECT_TOP_LEFT": (
        "Small rect. right top",
        "Small rect. right main",
    ),
    "RIGHT_SMALL_RECT_BOTTOM_LEFT": (
        "Small rect. right bottom",
        "Small rect. right main",
    ),
    "RIGHT_SMALL_RECT_TOP_RIGHT": (
        "Side line right",
        "Small rect. right top",
    ),
    "RIGHT_SMALL_RECT_BOTTOM_RIGHT": (
        "Side line right",
        "Small rect. right bottom",
    ),

    "CENTER_TOP": (
        "Middle line",
        "Side line top",
    ),
    "CENTER_BOTTOM": (
        "Middle line",
        "Side line bottom",
    ),
}

WORLD_KEYPOINTS = {
    "TL_PITCH_CORNER": (0.0, 0.0),
    "TR_PITCH_CORNER": (105.0, 0.0),
    "BL_PITCH_CORNER": (0.0, 68.0),
    "BR_PITCH_CORNER": (105.0, 68.0),

    "LEFT_BIG_RECT_TOP_LEFT": (0.0, 13.84),
    "LEFT_BIG_RECT_BOTTOM_LEFT": (0.0, 54.16),
    "LEFT_BIG_RECT_TOP_RIGHT": (16.5, 13.84),
    "LEFT_BIG_RECT_BOTTOM_RIGHT": (16.5, 54.16),

    "RIGHT_BIG_RECT_TOP_LEFT": (88.5, 13.84),
    "RIGHT_BIG_RECT_BOTTOM_LEFT": (88.5, 54.16),
    "RIGHT_BIG_RECT_TOP_RIGHT": (105.0, 13.84),
    "RIGHT_BIG_RECT_BOTTOM_RIGHT": (105.0, 54.16),

    "LEFT_SMALL_RECT_TOP_LEFT": (0.0, 24.84),
    "LEFT_SMALL_RECT_BOTTOM_LEFT": (0.0, 43.16),
    "LEFT_SMALL_RECT_TOP_RIGHT": (5.5, 24.84),
    "LEFT_SMALL_RECT_BOTTOM_RIGHT": (5.5, 43.16),

    "RIGHT_SMALL_RECT_TOP_LEFT": (99.5, 24.84),
    "RIGHT_SMALL_RECT_BOTTOM_LEFT": (99.5, 43.16),
    "RIGHT_SMALL_RECT_TOP_RIGHT": (105.0, 24.84),
    "RIGHT_SMALL_RECT_BOTTOM_RIGHT": (105.0, 43.16),

    "CENTER_TOP": (52.5, 0.0),
    "CENTER_BOTTOM": (52.5, 68.0),
}

def build_intersection_correspondences(
    predicted_lines,
    width,
    height,
):
    correspondences = []

    for keypoint_name, (line_a_name, line_b_name) in VALID_INTERSECTIONS.items():

        if line_a_name not in predicted_lines:
            continue

        if line_b_name not in predicted_lines:
            continue

        score_a = predicted_lines[line_a_name]["score"]
        score_b = predicted_lines[line_b_name]["score"]

#        if score_a < 0.7:
#            continue

#        if score_b < 0.7:
#            continue

        image_line_a = predicted_lines[line_a_name]["line"]
        image_line_b = predicted_lines[line_b_name]["line"]

        image_point = intersect_lines(
            image_line_a,
            image_line_b,
        )

        if image_point is None:
            continue

        if not point_inside_image(
            image_point,
            width,
            height,
        ):
            continue

        correspondences.append({
            "label": keypoint_name,
            "source": "semantic_keypoint",
            "img": image_point,
            "world": np.array(
                WORLD_KEYPOINTS[keypoint_name],
                dtype=np.float64,
            ),
        })

    return correspondences


def compute_homography(correspondences, ransac_threshold):
    if len(correspondences) < 4:
        return None, None

    img_pts = np.array(
        [item["img"] for item in correspondences],
        dtype=np.float32,
    )

    world_pts = np.array(
        [item["world"] for item in correspondences],
        dtype=np.float32,
    ) * SCALE

    H, mask = cv2.findHomography(
        img_pts,
        world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )

    return H, mask


def project_points(H, img_points):
    pts = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return projected.astype(np.float64)


def evaluate_homography(H, mask, correspondences):
    if H is None:
        return []

    img_pts = [item["img"] for item in correspondences]
    world_pts = np.array(
        [item["world"] for item in correspondences],
        dtype=np.float64,
    ) * SCALE

    projected = project_points(H, img_pts)
    errors = np.linalg.norm(projected - world_pts, axis=1)

    inliers = (
        mask.reshape(-1).astype(bool)
        if mask is not None
        else np.ones(len(errors), dtype=bool)
    )

    rows = []

    for item, error, is_inlier, projected_point, expected_point in zip(
        correspondences,
        errors,
        inliers,
        projected,
        world_pts,
    ):
        rows.append({
            "label": item["label"],
            "source": item["source"],
            "error_px": float(error),
            "inlier": bool(is_inlier),
            "img_x": float(item["img"][0]),
            "img_y": float(item["img"][1]),
            "expected_bev_x": float(expected_point[0]),
            "expected_bev_y": float(expected_point[1]),
            "projected_bev_x": float(projected_point[0]),
            "projected_bev_y": float(projected_point[1]),
        })

    return rows


def draw_infinite_line(image, line, color, thickness=2):
    height, width = image.shape[:2]

    a, b, c = line

    points = []

    if abs(b) > 1e-9:
        y0 = -(a * 0 + c) / b
        y1 = -(a * width + c) / b
        points.append((0, y0))
        points.append((width, y1))

    if abs(a) > 1e-9:
        x0 = -(b * 0 + c) / a
        x1 = -(b * height + c) / a
        points.append((x0, 0))
        points.append((x1, height))

    valid = []

    for x, y in points:
        if -1000 <= x <= width + 1000 and -1000 <= y <= height + 1000:
            valid.append((int(round(x)), int(round(y))))

    if len(valid) >= 2:
        cv2.line(image, valid[0], valid[1], color, thickness, cv2.LINE_AA)


def draw_debug_overlay(image, predicted_lines, correspondences, H, mask, output_path):
    overlay = image.copy()

#    for label, value in predicted_lines.items():
#        draw_infinite_line(
#            overlay,
#            value["line"],
#            color=(255, 0, 0),
#            thickness=2,
#        )

    inliers = (
        mask.reshape(-1).astype(bool)
        if mask is not None
        else np.zeros(len(correspondences), dtype=bool)
    )

    for item, is_inlier in zip(correspondences, inliers):
        point = item["img"]
        color = (0, 200, 0) if is_inlier else (0, 0, 255)

        cv2.circle(
            overlay,
            tuple(np.round(point).astype(int)),
            6,
            color,
            -1,
        )

        cv2.putText(
            overlay,
            item["label"],
            tuple(np.round(point + np.array([6, -6])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_image(model, image_path, output_dir, args):
    image, predicted_lines = predict_lines(
        model=model,
        image_path=image_path,
        conf=args.conf,
    )

    height, width = image.shape[:2]

    correspondences = build_intersection_correspondences(
        predicted_lines=predicted_lines,
        width=width,
        height=height,
    )

    H, mask = compute_homography(
        correspondences=correspondences,
        ransac_threshold=args.ransac_threshold,
    )

    error_rows = evaluate_homography(H, mask, correspondences)

    inlier_count = 0
    mean_inlier_error = np.nan
    max_inlier_error = np.nan

    if error_rows:
        inlier_errors = [
            row["error_px"]
            for row in error_rows
            if row["inlier"]
        ]

        inlier_count = len(inlier_errors)

        if inlier_errors:
            mean_inlier_error = float(np.mean(inlier_errors))
            max_inlier_error = float(np.max(inlier_errors))

    status = "ok"

    if H is None:
        status = "no_homography"
    elif inlier_count < args.min_inliers:
        status = "too_few_inliers"
    elif max_inlier_error > args.max_reprojection_error:
        status = "high_inlier_error"

    stem = image_path.stem

    overlay_path = output_dir / "debug_overlays" / f"{stem}_overlay.png"
    draw_debug_overlay(
        image=image,
        predicted_lines=predicted_lines,
        correspondences=correspondences,
        H=H,
        mask=mask,
        output_path=overlay_path,
    )

    if H is not None:
        bev = cv2.warpPerspective(
            image,
            H,
            (int(PITCH_WIDTH * SCALE), int(PITCH_HEIGHT * SCALE)),
        )
        cv2.imwrite(str(output_dir / "bev" / f"{stem}_bev.png"), bev)

    for row in error_rows:
        row["frame"] = image_path.name

    return {
        "frame": image_path.name,
        "status": status,
        "num_predicted_lines": len(predicted_lines),
        "num_correspondences": len(correspondences),
        "num_inliers": inlier_count,
        "mean_inlier_reprojection_error_px": mean_inlier_error,
        "max_inlier_reprojection_error_px": max_inlier_error,
    }, error_rows, H


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--max-reprojection-error", type=float, default=35.0)

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "debug_overlays").mkdir(parents=True, exist_ok=True)
    (output_dir / "bev").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    image_paths = sorted(
        list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.PNG"))
        + list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.JPG"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.JPEG"))
    )

    quality_rows = []
    point_error_rows = []
    homographies = {}

    for image_path in image_paths:
        print(f"Processing {image_path.name}")

        quality_row, error_rows, H = process_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            args=args,
        )

        quality_rows.append(quality_row)
        point_error_rows.extend(error_rows)

        if H is not None:
            homographies[image_path.name] = H

        print(
            f" -> {quality_row['status']} | "
            f"lines={quality_row['num_predicted_lines']} | "
            f"corr={quality_row['num_correspondences']} | "
            f"inliers={quality_row['num_inliers']}"
        )

    if homographies:
        np.savez(output_dir / "homographies.npz", **homographies)

    write_csv(
        output_dir / "homography_quality.csv",
        quality_rows,
        [
            "frame",
            "status",
            "num_predicted_lines",
            "num_correspondences",
            "num_inliers",
            "mean_inlier_reprojection_error_px",
            "max_inlier_reprojection_error_px",
        ],
    )

    write_csv(
        output_dir / "homography_point_errors.csv",
        point_error_rows,
        [
            "frame",
            "label",
            "source",
            "error_px",
            "inlier",
            "img_x",
            "img_y",
            "expected_bev_x",
            "expected_bev_y",
            "projected_bev_x",
            "projected_bev_y",
        ],
    )

    print()
    print("Done.")
    print(f"Output: {output_dir}")
    print(f"Quality CSV: {output_dir / 'homography_quality.csv'}")
    print(f"Point errors: {output_dir / 'homography_point_errors.csv'}")
    print(f"Overlays: {output_dir / 'debug_overlays'}")
    print(f"BEV: {output_dir / 'bev'}")


if __name__ == "__main__":
    main()