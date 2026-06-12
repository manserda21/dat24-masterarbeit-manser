import json
import argparse
from pathlib import Path
from collections import defaultdict


def point_key(world_point, decimals=3):
    """
    Erzeugt einen Hash für Weltkoordinaten.
    Rundung verhindert Floating-Point Probleme.
    """
    return (
        round(float(world_point[0]), decimals),
        round(float(world_point[1]), decimals),
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True,
        help="calibration_dataset.json",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    json_path = Path(args.json)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        dataset = json.load(f)

    unique_points = {}
    occurrences = defaultdict(int)
    source_names = defaultdict(set)

    next_id = 0

    for frame in dataset:

        for point_name, point_data in frame["all_points"].items():

            world = point_data["world"]

            key = point_key(world)

            occurrences[key] += 1
            source_names[key].add(point_name)

            if key not in unique_points:

                unique_points[key] = {
                    "id": f"kp_{next_id:03d}",
                    "world_x": float(world[0]),
                    "world_y": float(world[1]),
                }

                next_id += 1

    catalog = []

    for key, point_info in unique_points.items():

        catalog.append({
            "keypoint_id": point_info["id"],
            "world_x": point_info["world_x"],
            "world_y": point_info["world_y"],
            "count": occurrences[key],
            "source_labels": sorted(list(source_names[key])),
        })

    catalog = sorted(
        catalog,
        key=lambda x: x["keypoint_id"]
    )

    output_json = output_dir / "unique_keypoints.json"

    with open(output_json, "w") as f:
        json.dump(catalog, f, indent=2)

    print()
    print("=" * 40)
    print("UNIQUE KEYPOINT ANALYSIS")
    print("=" * 40)
    print(f"Unique keypoints: {len(catalog)}")
    print(f"Output: {output_json}")
    print()

    for kp in catalog:
        print(
            f"{kp['keypoint_id']:8s} "
            f"({kp['world_x']:6.2f}, {kp['world_y']:6.2f}) "
            f"count={kp['count']}"
        )


if __name__ == "__main__":
    main()