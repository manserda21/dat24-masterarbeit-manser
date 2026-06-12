import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def rounded_key(world, decimals=3):
    return (
        round(float(world[0]), decimals),
        round(float(world[1]), decimals),
    )


def load_keypoint_catalog(path):
    with open(path, "r") as f:
        catalog = json.load(f)

    key_to_id = {}
    id_to_world = {}

    for kp in catalog:
        kp_id = kp["keypoint_id"]
        world = (round(float(kp["world_x"]), 3), round(float(kp["world_y"]), 3))

        key_to_id[world] = kp_id
        id_to_world[kp_id] = {
            "world_x": float(kp["world_x"]),
            "world_y": float(kp["world_y"]),
            "count": int(kp["count"]),
            "source_labels": kp.get("source_labels", []),
        }

    return key_to_id, id_to_world


def draw_gaussian(heatmap, x, y, sigma):
    height, width = heatmap.shape

    radius = int(3 * sigma)

    x0 = max(0, int(round(x)) - radius)
    x1 = min(width - 1, int(round(x)) + radius)

    y0 = max(0, int(round(y)) - radius)
    y1 = min(height - 1, int(round(y)) + radius)

    if x0 > x1 or y0 > y1:
        return

    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)

    xx, yy = np.meshgrid(xs, ys)

    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(
        heatmap[y0:y1 + 1, x0:x1 + 1],
        gaussian,
    )


def create_heatmaps(record, key_to_id, kp_id_to_index, heatmap_width, heatmap_height, sigma):
    num_keypoints = len(kp_id_to_index)
    heatmaps = np.zeros(
        (num_keypoints, heatmap_height, heatmap_width),
        dtype=np.float32,
    )

    visible_points = []

    image_width = float(record["width"])
    image_height = float(record["height"])

    scale_x = heatmap_width / image_width
    scale_y = heatmap_height / image_height

    for point_name, point_data in record["all_points"].items():
        world_key = rounded_key(point_data["world"])

        if world_key not in key_to_id:
            continue

        kp_id = key_to_id[world_key]
        kp_index = kp_id_to_index[kp_id]

        x_img, y_img = point_data["image"]

        x_hm = float(x_img) * scale_x
        y_hm = float(y_img) * scale_y

        draw_gaussian(
            heatmap=heatmaps[kp_index],
            x=x_hm,
            y=y_hm,
            sigma=sigma,
        )

        visible_points.append({
            "keypoint_id": kp_id,
            "keypoint_index": kp_index,
            "image_x": float(x_img),
            "image_y": float(y_img),
            "heatmap_x": float(x_hm),
            "heatmap_y": float(y_hm),
            "world_x": float(point_data["world"][0]),
            "world_y": float(point_data["world"][1]),
            "source": point_data.get("source", "unknown"),
            "source_label": point_name,
        })

    return heatmaps, visible_points


def prepare_dataset(
    json_path,
    keypoints_path,
    images_dir,
    output_dir,
    val_ratio,
    seed,
    heatmap_width,
    heatmap_height,
    sigma,
):
    with open(json_path, "r") as f:
        records = json.load(f)

    records = [
        record for record in records
        if len(record.get("all_points", {})) > 0
    ]

    key_to_id, id_to_world = load_keypoint_catalog(keypoints_path)

    keypoint_ids = sorted(id_to_world.keys())
    kp_id_to_index = {
        kp_id: index
        for index, kp_id in enumerate(keypoint_ids)
    }

    random.seed(seed)
    random.shuffle(records)

    n_val = max(1, int(len(records) * val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    metadata = {
        "num_keypoints": len(keypoint_ids),
        "heatmap_width": heatmap_width,
        "heatmap_height": heatmap_height,
        "sigma": sigma,
        "keypoints": {},
    }

    for kp_id in keypoint_ids:
        metadata["keypoints"][kp_id] = {
            "index": kp_id_to_index[kp_id],
            **id_to_world[kp_id],
        }

    with open(output_dir / "keypoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    split_summary = {}

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
    ]:
        image_out_dir = output_dir / "images" / split_name
        heatmap_out_dir = output_dir / "heatmaps" / split_name
        points_out_dir = output_dir / "points" / split_name

        image_out_dir.mkdir(parents=True, exist_ok=True)
        heatmap_out_dir.mkdir(parents=True, exist_ok=True)
        points_out_dir.mkdir(parents=True, exist_ok=True)

        split_summary[split_name] = {
            "images": len(split_records),
            "points": 0,
        }

        for record in split_records:
            image_name = record["image"]
            stem = Path(image_name).stem

            src_image = images_dir / image_name
            dst_image = image_out_dir / image_name

            if not src_image.exists():
                print(f"Missing image: {src_image}")
                continue

            shutil.copy2(src_image, dst_image)

            heatmaps, visible_points = create_heatmaps(
                record=record,
                key_to_id=key_to_id,
                kp_id_to_index=kp_id_to_index,
                heatmap_width=heatmap_width,
                heatmap_height=heatmap_height,
                sigma=sigma,
            )

            np.savez_compressed(
                heatmap_out_dir / f"{stem}.npz",
                heatmaps=heatmaps,
            )

            with open(points_out_dir / f"{stem}.json", "w") as f:
                json.dump(visible_points, f, indent=2)

            split_summary[split_name]["points"] += len(visible_points)

    config = {
        "path": str(output_dir.resolve()),
        "train_images": "images/train",
        "val_images": "images/val",
        "train_heatmaps": "heatmaps/train",
        "val_heatmaps": "heatmaps/val",
        "train_points": "points/train",
        "val_points": "points/val",
        "metadata": "keypoint_metadata.json",
        "num_keypoints": len(keypoint_ids),
        "heatmap_size": [heatmap_width, heatmap_height],
    }

    with open(output_dir / "dataset_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print()
    print("=" * 40)
    print("PITCH KEYPOINT HEATMAP DATASET")
    print("=" * 40)
    print(f"Input records:       {len(records)}")
    print(f"Train images:        {split_summary['train']['images']}")
    print(f"Val images:          {split_summary['val']['images']}")
    print(f"Train points:        {split_summary['train']['points']}")
    print(f"Val points:          {split_summary['val']['points']}")
    print(f"Keypoints:           {len(keypoint_ids)}")
    print(f"Heatmap size:        {heatmap_width} x {heatmap_height}")
    print(f"Output:              {output_dir}")
    print(f"Config:              {output_dir / 'dataset_config.yaml'}")
    print()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json", required=True)
    parser.add_argument("--keypoints", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap-width", type=int, default=320)
    parser.add_argument("--heatmap-height", type=int, default=180)
    parser.add_argument("--sigma", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_dataset(
        json_path=Path(args.json),
        keypoints_path=Path(args.keypoints),
        images_dir=Path(args.images_dir),
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        heatmap_width=args.heatmap_width,
        heatmap_height=args.heatmap_height,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()