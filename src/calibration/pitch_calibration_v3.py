# calibration/pitch_calibration_v3.py

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0
SCALE = 10.0

WORLD_POINTS = {
    "TL_PITCH_CORNER": (0.0, 0.0),
    "TR_PITCH_CORNER": (PITCH_WIDTH, 0.0),
    "BL_PITCH_CORNER": (0.0, PITCH_HEIGHT),
    "BR_PITCH_CORNER": (PITCH_WIDTH, PITCH_HEIGHT),
    "CENTER_POINT": (PITCH_WIDTH / 2.0, PITCH_HEIGHT / 2.0),
    "PENALTY_SPOT_LEFT": (11.0, PITCH_HEIGHT / 2.0),
    "PENALTY_SPOT_RIGHT": (PITCH_WIDTH - 11.0, PITCH_HEIGHT / 2.0),
    "CENTER_CIRCLE_TOP": (PITCH_WIDTH / 2.0, (PITCH_HEIGHT / 2.0) - 9.15),
    "CENTER_CIRCLE_BOTTOM": (PITCH_WIDTH / 2.0, (PITCH_HEIGHT / 2.0) + 9.15),
    "CENTER_CIRCLE_LEFT": ((PITCH_WIDTH / 2.0) - 9.15, PITCH_HEIGHT / 2.0),
    "CENTER_CIRCLE_RIGHT": ((PITCH_WIDTH / 2.0) + 9.15, PITCH_HEIGHT / 2.0),
}

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

LINE_ALIASES = {
    "touchline_top": "Side line top",
    "touchline_bottom": "Side line bottom",
    "goal_line_left": "Side line left",
    "goal_line_right": "Side line right",
    "center_line": "Middle line",
    "middle_line": "Middle line",
    "halfway_line": "Middle line",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute pitch homographies with quality metrics and optional line intersections."
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ransac_threshold", type=float, default=3.0)
    parser.add_argument("--max_reprojection_error", type=float, default=25.0)
    parser.add_argument("--min_points", type=int, default=4)
    parser.add_argument("--min_inliers", type=int, default=6)
    parser.add_argument("--no_line_intersections", action="store_true")
    parser.add_argument(
        "--save_all_homographies",
        action="store_true",
        help="Save homographies even when quality checks fail.",
    )
    parser.add_argument(
        "--exclude_labels",
        type=str,
        default="",
        help="Comma-separated point or generated intersection labels to ignore.",
    )
    return parser.parse_args()


def normalize_label(label):
    cleaned = re.sub(r"\s+", " ", label.strip())
    return LINE_ALIASES.get(cleaned, cleaned)


def parse_excluded_labels(value):
    return {
        label.strip()
        for label in value.split(",")
        if label.strip()
    }


def parse_points_attr(value):
    points = []
    for pair in value.split(";"):
        x, y = map(float, pair.split(","))
        points.append((x, y))
    return points


def load_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()
    frames = {}

    for image in root.findall("image"):
        name = image.get("name")
        points = []
        lines = {}

        for point_node in image.findall("points"):
            label = point_node.get("label")
            coords = parse_points_attr(point_node.get("points"))
            if coords:
                points.append((label, coords[0]))

        for polyline_node in image.findall("polyline"):
            label = normalize_label(polyline_node.get("label"))
            coords = parse_points_attr(polyline_node.get("points"))
            if len(coords) >= 2:
                lines[label] = coords

        frames[name] = {
            "points": points,
            "lines": lines,
        }

    return frames


def homogeneous_line_from_points(p1, p2):
    hp1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    hp2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
    line = np.cross(hp1, hp2)
    norm = np.linalg.norm(line[:2])

    if norm > 0:
        line = line / norm

    return line


def fit_image_line(points):
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    direction = vh[0]
    p1 = centroid - direction
    p2 = centroid + direction
    return homogeneous_line_from_points(p1, p2)


def intersect_lines(line_a, line_b):
    point = np.cross(line_a, line_b)

    if abs(point[2]) < 1e-9:
        return None

    return np.array([point[0] / point[2], point[1] / point[2]], dtype=np.float64)


def point_near_polyline_bbox(point, polyline_a, polyline_b, margin=80.0):
    x, y = point

    for polyline in (polyline_a, polyline_b):
        pts = np.asarray(polyline, dtype=np.float64)
        min_x, min_y = pts.min(axis=0) - margin
        max_x, max_y = pts.max(axis=0) + margin

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

    return True


def build_point_correspondences(frame, excluded_labels):
    correspondences = []

    for label, img_point in frame["points"]:
        if label not in WORLD_POINTS:
            continue

        if label in excluded_labels:
            continue

        correspondences.append({
            "label": label,
            "source": "point",
            "img": np.array(img_point, dtype=np.float64),
            "world": np.array(WORLD_POINTS[label], dtype=np.float64),
        })

    return correspondences


def build_line_intersection_correspondences(frame, excluded_labels):
    recognized_lines = {
        label: coords
        for label, coords in frame["lines"].items()
        if label in PITCH_LINES
    }
    correspondences = []

    for label_a, label_b in combinations(sorted(recognized_lines), 2):
        image_line_a = fit_image_line(recognized_lines[label_a])
        image_line_b = fit_image_line(recognized_lines[label_b])
        image_point = intersect_lines(image_line_a, image_line_b)

        if image_point is None:
            continue

        if not point_near_polyline_bbox(
            image_point,
            recognized_lines[label_a],
            recognized_lines[label_b],
        ):
            continue

        world_line_a = homogeneous_line_from_points(*PITCH_LINES[label_a])
        world_line_b = homogeneous_line_from_points(*PITCH_LINES[label_b])
        world_point = intersect_lines(world_line_a, world_line_b)

        if world_point is None:
            continue

        if not (0.0 <= world_point[0] <= PITCH_WIDTH and 0.0 <= world_point[1] <= PITCH_HEIGHT):
            continue

        intersection_label = f"{label_a} x {label_b}"

        if intersection_label in excluded_labels:
            continue

        correspondences.append({
            "label": intersection_label,
            "source": "line_intersection",
            "img": image_point,
            "world": world_point,
        })

    return correspondences


def build_correspondences(frame, use_line_intersections, excluded_labels):
    correspondences = build_point_correspondences(frame, excluded_labels)

    if use_line_intersections:
        correspondences.extend(build_line_intersection_correspondences(frame, excluded_labels))

    return correspondences


def project_points(H, img_points):
    pts = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return projected.astype(np.float64)


def compute_homography(correspondences, args):
    if len(correspondences) < args.min_points:
        return None, None

    img_pts = np.array([item["img"] for item in correspondences], dtype=np.float32)
    world_pts = np.array([item["world"] for item in correspondences], dtype=np.float32) * SCALE

    H, mask = cv2.findHomography(
        img_pts,
        world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=args.ransac_threshold,
    )

    return H, mask


def evaluate_homography(H, mask, correspondences):
    img_pts = [item["img"] for item in correspondences]
    world_pts = np.array([item["world"] for item in correspondences], dtype=np.float64) * SCALE
    projected = project_points(H, img_pts)
    errors = np.linalg.norm(projected - world_pts, axis=1)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(errors), dtype=bool)

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


def draw_debug_overlay(image, H, mask, correspondences, output_path):
    overlay = image.copy()
    inverse_H = np.linalg.inv(H)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(correspondences), dtype=bool)

    for item, is_inlier in zip(correspondences, inliers):
        img_point = item["img"]
        expected_bev = item["world"] * SCALE
        expected_img = project_points(inverse_H, [expected_bev])[0]
        color = (0, 180, 0) if is_inlier else (0, 0, 255)

        cv2.circle(overlay, tuple(np.round(img_point).astype(int)), 5, color, -1)
        cv2.circle(overlay, tuple(np.round(expected_img).astype(int)), 5, (255, 0, 0), 2)
        cv2.line(
            overlay,
            tuple(np.round(img_point).astype(int)),
            tuple(np.round(expected_img).astype(int)),
            (0, 255, 255),
            1,
        )
        cv2.putText(
            overlay,
            item["source"][0],
            tuple(np.round(img_point + np.array([6, -6])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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


def main():
    args = parse_arguments()
    dataset_path = Path(args.dataset_path)
    annotations_path = Path(args.annotations) if args.annotations else dataset_path / "annotations.xml"
    images_dir = Path(args.images_dir) if args.images_dir else dataset_path / "images"
    output_dir = Path(args.output_dir)
    overlays_dir = output_dir / "debug_overlays"
    bev_dir = output_dir / "bev"

    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    bev_dir.mkdir(parents=True, exist_ok=True)

    frames = load_annotations(annotations_path)
    homographies = {}
    quality_rows = []
    point_error_rows = []
    unknown_line_labels = set()
    use_line_intersections = not args.no_line_intersections
    excluded_labels = parse_excluded_labels(args.exclude_labels)

    if excluded_labels:
        print("\nExcluded labels:")
        for label in sorted(excluded_labels):
            print(f"  {label}")

    for name, frame in frames.items():
        image_path = images_dir / name
        print(f"\nProcessing: {name}")

        for line_label in frame["lines"]:
            if line_label not in PITCH_LINES:
                unknown_line_labels.add(line_label)

        image = cv2.imread(str(image_path))
        if image is None:
            print(" -> image not found")
            quality_rows.append({"frame": name, "status": "missing_image"})
            continue

        correspondences = build_correspondences(
            frame,
            use_line_intersections,
            excluded_labels,
        )
        point_count = sum(item["source"] == "point" for item in correspondences)
        intersection_count = sum(item["source"] == "line_intersection" for item in correspondences)

        H, mask = compute_homography(correspondences, args)

        if H is None:
            print(" -> skipped (not enough correspondences)")
            quality_rows.append({
                "frame": name,
                "status": "not_enough_correspondences",
                "num_correspondences": len(correspondences),
                "num_points": point_count,
                "num_line_intersections": intersection_count,
            })
            continue

        error_rows = evaluate_homography(H, mask, correspondences)
        errors = np.array([row["error_px"] for row in error_rows], dtype=np.float64)
        inlier_errors = np.array(
            [row["error_px"] for row in error_rows if row["inlier"]],
            dtype=np.float64,
        )
        inlier_count = sum(row["inlier"] for row in error_rows)
        mean_error = float(errors.mean()) if len(errors) else np.nan
        max_error = float(errors.max()) if len(errors) else np.nan
        mean_inlier_error = float(inlier_errors.mean()) if len(inlier_errors) else np.nan
        max_inlier_error = float(inlier_errors.max()) if len(inlier_errors) else np.nan

        if inlier_count < args.min_inliers:
            status = "too_few_inliers"
        elif max_inlier_error > args.max_reprojection_error:
            status = "high_inlier_error"
        else:
            status = "ok"

        for row in error_rows:
            row["frame"] = name
            point_error_rows.append(row)

        if status == "ok" or args.save_all_homographies:
            homographies[name] = H

        warped = cv2.warpPerspective(
            image,
            H,
            (int(PITCH_WIDTH * SCALE), int(PITCH_HEIGHT * SCALE)),
        )
        cv2.imwrite(str(bev_dir / f"{Path(name).stem}_bev.png"), warped)
        draw_debug_overlay(image, H, mask, correspondences, overlays_dir / f"{Path(name).stem}_overlay.png")

        quality_rows.append({
            "frame": name,
            "status": status,
            "num_correspondences": len(correspondences),
            "num_points": point_count,
            "num_line_intersections": intersection_count,
            "num_inliers": inlier_count,
            "mean_reprojection_error_px": mean_error,
            "max_reprojection_error_px": max_error,
            "mean_inlier_reprojection_error_px": mean_inlier_error,
            "max_inlier_reprojection_error_px": max_inlier_error,
            "used_labels": ";".join(item["label"] for item in correspondences),
        })

        print(
            " -> saved "
            f"({len(correspondences)} correspondences, "
            f"{inlier_count} inliers, "
            f"mean inlier error {mean_inlier_error:.2f}px, "
            f"mean all error {mean_error:.2f}px)"
        )

    if homographies:
        np.savez(output_dir / "homographies.npz", **homographies)

    write_csv(
        output_dir / "homography_quality.csv",
        quality_rows,
        [
            "frame",
            "status",
            "num_correspondences",
            "num_points",
            "num_line_intersections",
            "num_inliers",
            "mean_reprojection_error_px",
            "max_reprojection_error_px",
            "mean_inlier_reprojection_error_px",
            "max_inlier_reprojection_error_px",
            "used_labels",
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

    if unknown_line_labels:
        print("\nUnknown line labels not used for intersections:")
        for label in sorted(unknown_line_labels):
            print(f"  {label}")

    print("\nDone.")
    print(f"Homographies: {output_dir / 'homographies.npz'}")
    print(f"Quality CSV:  {output_dir / 'homography_quality.csv'}")
    print(f"Point errors: {output_dir / 'homography_point_errors.csv'}")
    print(f"Overlays:     {overlays_dir}")


if __name__ == "__main__":
    main()
