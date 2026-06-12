import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


PITCH_WIDTH = 105.0
PITCH_HEIGHT = 68.0

POINT_WORLD = {
    "TL_PITCH_CORNER": [0.0, 0.0],
    "TR_PITCH_CORNER": [PITCH_WIDTH, 0.0],
    "BL_PITCH_CORNER": [0.0, PITCH_HEIGHT],
    "BR_PITCH_CORNER": [PITCH_WIDTH, PITCH_HEIGHT],
    "CENTER_POINT": [PITCH_WIDTH / 2.0, PITCH_HEIGHT / 2.0],
    "PENALTY_SPOT_LEFT": [11.0, PITCH_HEIGHT / 2.0],
    "PENALTY_SPOT_RIGHT": [PITCH_WIDTH - 11.0, PITCH_HEIGHT / 2.0],
    "CENTER_CIRCLE_TOP": [PITCH_WIDTH / 2.0, PITCH_HEIGHT / 2.0 - 9.15],
    "CENTER_CIRCLE_BOTTOM": [PITCH_WIDTH / 2.0, PITCH_HEIGHT / 2.0 + 9.15],
    "CENTER_CIRCLE_LEFT": [PITCH_WIDTH / 2.0 - 9.15, PITCH_HEIGHT / 2.0],
    "CENTER_CIRCLE_RIGHT": [PITCH_WIDTH / 2.0 + 9.15, PITCH_HEIGHT / 2.0],
}

LINE_WORLD = {
    "Side line top": [[0.0, 0.0], [PITCH_WIDTH, 0.0]],
    "Side line bottom": [[0.0, PITCH_HEIGHT], [PITCH_WIDTH, PITCH_HEIGHT]],
    "Side line left": [[0.0, 0.0], [0.0, PITCH_HEIGHT]],
    "Side line right": [[PITCH_WIDTH, 0.0], [PITCH_WIDTH, PITCH_HEIGHT]],
    "Middle line": [[PITCH_WIDTH / 2.0, 0.0], [PITCH_WIDTH / 2.0, PITCH_HEIGHT]],

    "Big rect. left top": [[0.0, 13.84], [16.5, 13.84]],
    "Big rect. left bottom": [[0.0, 54.16], [16.5, 54.16]],
    "Big rect. left main": [[16.5, 13.84], [16.5, 54.16]],

    "Small rect. left top": [[0.0, 24.84], [5.5, 24.84]],
    "Small rect. left bottom": [[0.0, 43.16], [5.5, 43.16]],
    "Small rect. left main": [[5.5, 24.84], [5.5, 43.16]],

    "Big rect. right top": [[PITCH_WIDTH - 16.5, 13.84], [PITCH_WIDTH, 13.84]],
    "Big rect. right bottom": [[PITCH_WIDTH - 16.5, 54.16], [PITCH_WIDTH, 54.16]],
    "Big rect. right main": [[PITCH_WIDTH - 16.5, 13.84], [PITCH_WIDTH - 16.5, 54.16]],

    "Small rect. right top": [[PITCH_WIDTH - 5.5, 24.84], [PITCH_WIDTH, 24.84]],
    "Small rect. right bottom": [[PITCH_WIDTH - 5.5, 43.16], [PITCH_WIDTH, 43.16]],
    "Small rect. right main": [[PITCH_WIDTH - 5.5, 24.84], [PITCH_WIDTH - 5.5, 43.16]],
}


def parse_points(value):
    points = []

    for pair in value.split(";"):
        x_str, y_str = pair.split(",")
        points.append([float(x_str), float(y_str)])

    return points


def line_from_two_points(p1, p2):
    p1_h = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    p2_h = np.array([p2[0], p2[1], 1.0], dtype=np.float64)

    line = np.cross(p1_h, p2_h)
    norm = np.linalg.norm(line[:2])

    if norm > 0:
        line = line / norm

    return line


def fit_line(points):
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) < 2:
        return None

    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)

    direction = vh[0]

    p1 = centroid - direction
    p2 = centroid + direction

    return line_from_two_points(p1, p2)


def intersect_lines(line_a, line_b):
    point = np.cross(line_a, line_b)

    if abs(point[2]) < 1e-9:
        return None

    return np.array(
        [
            point[0] / point[2],
            point[1] / point[2],
        ],
        dtype=np.float64,
    )


def is_inside_pitch(point):
    x, y = point

    return (
        0.0 <= x <= PITCH_WIDTH
        and 0.0 <= y <= PITCH_HEIGHT
    )


def is_near_visible_segments(point, line_a_points, line_b_points, margin=120.0):
    x, y = point

    for pts in [line_a_points, line_b_points]:
        arr = np.asarray(pts, dtype=np.float64)

        min_x, min_y = arr.min(axis=0) - margin
        max_x, max_y = arr.max(axis=0) + margin

        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False

    return True


def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames = []

    for image_node in root.findall("image"):
        image_name = image_node.get("name")
        width = int(image_node.get("width"))
        height = int(image_node.get("height"))

        points = {}
        lines = {}

        for point_node in image_node.findall("points"):
            label = point_node.get("label")

            if label not in POINT_WORLD:
                continue

            coords = parse_points(point_node.get("points"))

            if not coords:
                continue

            points[label] = {
                "image": coords[0],
                "world": POINT_WORLD[label],
            }

        for line_node in image_node.findall("polyline"):
            label = line_node.get("label")

            if label not in LINE_WORLD:
                continue

            coords = parse_points(line_node.get("points"))

            if len(coords) < 2:
                continue

            if label not in lines:
                lines[label] = []

            lines[label].append({
                "image": coords,
                "world": LINE_WORLD[label],
            })

        frames.append({
            "image": image_name,
            "width": width,
            "height": height,
            "points": points,
            "lines": lines,
        })

    return frames


def generate_line_intersections(frame):
    generated = {}

    line_candidates = []

    for label, instances in frame["lines"].items():
        for instance_index, instance in enumerate(instances):
            image_line = fit_line(instance["image"])

            if image_line is None:
                continue

            world_line = line_from_two_points(
                instance["world"][0],
                instance["world"][1],
            )

            line_candidates.append({
                "label": label,
                "instance_index": instance_index,
                "image_points": instance["image"],
                "image_line": image_line,
                "world_line": world_line,
            })

    for line_a, line_b in combinations(line_candidates, 2):
        if line_a["label"] == line_b["label"]:
            continue

        image_intersection = intersect_lines(
            line_a["image_line"],
            line_b["image_line"],
        )

        world_intersection = intersect_lines(
            line_a["world_line"],
            line_b["world_line"],
        )

        if image_intersection is None or world_intersection is None:
            continue

        if not is_inside_pitch(world_intersection):
            continue

        if not is_near_visible_segments(
            image_intersection,
            line_a["image_points"],
            line_b["image_points"],
        ):
            continue

        name = (
            f"{line_a['label']} x {line_b['label']}"
            f" #{line_a['instance_index']}_{line_b['instance_index']}"
        )

        generated[name] = {
            "image": [
                float(image_intersection[0]),
                float(image_intersection[1]),
            ],
            "world": [
                float(world_intersection[0]),
                float(world_intersection[1]),
            ],
            "source": "line_intersection",
            "line_a": line_a["label"],
            "line_b": line_b["label"],
        }

    return generated


def build_dataset(frames):
    dataset = []

    for frame in frames:
        generated_points = generate_line_intersections(frame)

        direct_points = {
            label: {
                "image": value["image"],
                "world": value["world"],
                "source": "manual_point",
            }
            for label, value in frame["points"].items()
        }

        all_points = {}
        all_points.update(direct_points)
        all_points.update(generated_points)

        frame_record = {
            "image": frame["image"],
            "width": frame["width"],
            "height": frame["height"],
            "direct_points": direct_points,
            "generated_points": generated_points,
            "all_points": all_points,
            "lines": frame["lines"],
            "num_direct_points": len(direct_points),
            "num_generated_points": len(generated_points),
            "num_all_points": len(all_points),
            "num_line_labels": len(frame["lines"]),
        }

        dataset.append(frame_record)

    return dataset


def draw_debug_overlay(image_path, record, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return

    overlay = image.copy()

    for label, value in record["direct_points"].items():
        x, y = value["image"]

        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            6,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            overlay,
            label,
            (int(round(x)) + 5, int(round(y)) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for label, value in record["generated_points"].items():
        x, y = value["image"]

        cv2.circle(
            overlay,
            (int(round(x)), int(round(y))),
            5,
            (0, 0, 255),
            -1,
        )

    for label, instances in record["lines"].items():
        for instance in instances:
            pts = instance["image"]

            for i in range(1, len(pts)):
                p1 = pts[i - 1]
                p2 = pts[i]

                cv2.line(
                    overlay,
                    (int(round(p1[0])), int(round(p1[1]))),
                    (int(round(p2[0])), int(round(p2[1]))),
                    (255, 0, 0),
                    2,
                )

    cv2.imwrite(str(output_path), overlay)


def write_summary_csv(dataset, output_path):
    fieldnames = [
        "image",
        "num_direct_points",
        "num_generated_points",
        "num_all_points",
        "num_line_labels",
    ]

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record in dataset:
            writer.writerow({
                "image": record["image"],
                "num_direct_points": record["num_direct_points"],
                "num_generated_points": record["num_generated_points"],
                "num_all_points": record["num_all_points"],
                "num_line_labels": record["num_line_labels"],
            })


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing annotations.xml and images/",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--debug-overlays",
        action="store_true",
        help="Create debug overlay images",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    xml_path = dataset_dir / "annotations.xml"
    images_dir = dataset_dir / "images"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML file: {xml_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    frames = parse_cvat_xml(xml_path)
    dataset = build_dataset(frames)

    annotated_dataset = [
        record
        for record in dataset
        if record["num_all_points"] > 0 or record["num_line_labels"] > 0
    ]

    output_json = output_dir / "calibration_dataset.json"
    output_summary = output_dir / "calibration_summary.csv"

    with open(output_json, "w") as json_file:
        json.dump(annotated_dataset, json_file, indent=2)

    write_summary_csv(
        annotated_dataset,
        output_summary,
    )

    if args.debug_overlays:
        overlay_dir = output_dir / "debug_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        for record in annotated_dataset:
            image_path = images_dir / record["image"]
            output_path = overlay_dir / record["image"]

            draw_debug_overlay(
                image_path=image_path,
                record=record,
                output_path=output_path,
            )

    print()
    print("==============================")
    print("CALIBRATION DATASET CREATED")
    print("==============================")
    print(f"Frames in XML:        {len(frames)}")
    print(f"Annotated frames:     {len(annotated_dataset)}")
    print(f"Output JSON:          {output_json}")
    print(f"Summary CSV:          {output_summary}")

    if args.debug_overlays:
        print(f"Debug overlays:       {output_dir / 'debug_overlays'}")

    print()


if __name__ == "__main__":
    main()