# src/calibration/create_malaga_home_dataset.py

import json
import shutil
from pathlib import Path

import pandas as pd


SOURCE_BASE = Path(
    "/data/manser/soccernet/calibration-2023"
)

OUTPUT_BASE = Path(
    "/data/manser/datasets/calibration/malaga_home_stadium_v1"
)

MATCHES = {
    "spain_laliga-2016-2017- Malaga - Real Madrid-17-05-21":
        "malaga_real_2017",
    "spain_laliga-2015-2016- Malaga - Real Madrid-16-02-21":
        "malaga_real_2016",
    "spain_laliga-2016-2017- Malaga - Barcelona-17-04-08":
        "malaga_barca_2017",
}


def main():

    images_dir = OUTPUT_BASE / "images"
    annotations_dir = OUTPUT_BASE / "annotations"

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotations_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    for split in [
        "train",
        "valid",
        "test",
    ]:

        per_match_file = (
            SOURCE_BASE
            / split
            / "per_match_info.json"
        )

        if not per_match_file.exists():
            continue

        with open(per_match_file) as f:
            per_match = json.load(f)

        for match_name, prefix in MATCHES.items():

            if match_name not in per_match:
                continue

            num_images = len(
                per_match[match_name]
            )

            print(
                f"\nProcessing: {match_name}"
            )

            print(
                f"Frames: {num_images}"
            )

            for image_name in per_match[match_name]:

                image_id = Path(
                    image_name
                ).stem

                new_name = (
                    f"{prefix}_{image_id}"
                )

                source_image = (
                    SOURCE_BASE
                    / split
                    / image_name
                )

                source_json = (
                    SOURCE_BASE
                    / split
                    / f"{image_id}.json"
                )

                if not source_image.exists():

                    print(
                        f"Missing image: "
                        f"{source_image}"
                    )

                    continue

                if not source_json.exists():

                    print(
                        f"Missing json: "
                        f"{source_json}"
                    )

                    continue

                target_image = (
                    images_dir
                    / f"{new_name}.jpg"
                )

                target_json = (
                    annotations_dir
                    / f"{new_name}.json"
                )

                shutil.copy2(
                    source_image,
                    target_image,
                )

                shutil.copy2(
                    source_json,
                    target_json,
                )

                metadata_rows.append(
                    {
                        "image_name":
                            f"{new_name}.jpg",
                        "annotation_name":
                            f"{new_name}.json",
                        "match":
                            match_name,
                        "split":
                            split,
                        "prefix":
                            prefix,
                    }
                )

    metadata = pd.DataFrame(
        metadata_rows
    )

    metadata = metadata.sort_values(
        [
            "prefix",
            "image_name",
        ]
    )

    metadata.to_csv(
        OUTPUT_BASE / "metadata.csv",
        index=False,
    )

    print("\nDone")

    print(
        f"Images: {len(metadata)}"
    )

    print("\nImages per match:")

    print(
        metadata["prefix"]
        .value_counts()
    )


if __name__ == "__main__":
    main()