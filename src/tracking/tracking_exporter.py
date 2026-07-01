import pandas as pd


def export_tracking_to_parquet(
    results,
    output_path: str,
    fps: float,
):
    """
    Export tracking results to parquet.

    One row corresponds to one tracked
    detection in one frame.
    """

    rows = []

    for frame_idx, result in enumerate(results):

        if result.boxes is None:
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            cls_id = int(
                boxes.cls[i].item()
            )

            conf = float(
                boxes.conf[i].item()
            )

            track_id = None

            if (
                boxes.id is not None
                and len(boxes.id) > i
            ):
                track_id = int(
                    boxes.id[i].item()
                )

            class_name = result.names[
                cls_id
            ]

            team = (
                "opponent"
                if class_name == "opponent"
                else "real_madrid"
            )

            x_center, y_center, width, height = (
                boxes.xywh[i]
                .cpu()
                .numpy()
            )

            x_min, y_min, x_max, y_max = (
                boxes.xyxy[i]
                .cpu()
                .numpy()
            )

            rows.append(
                {
                    "frame": frame_idx,
                    "time_seconds": frame_idx / fps,
                    "time_ms": int(
                        (frame_idx / fps) * 1000
                    ),
                    "fps": fps,

                    "object_type": "player",
                    "team": team,

                    "track_id": track_id,

                    "class_id": cls_id,
                    "class_name": class_name,

                    "confidence": conf,

                    "x_center": float(
                        x_center
                    ),
                    "y_center": float(
                        y_center
                    ),

                    "width": float(
                        width
                    ),
                    "height": float(
                        height
                    ),

                    "x_min": float(
                        x_min
                    ),
                    "y_min": float(
                        y_min
                    ),
                    "x_max": float(
                        x_max
                    ),
                    "y_max": float(
                        y_max
                    ),

                    "source_model": "player_model",
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:

        print(
            "No tracking detections found."
        )

        return

    df = df.astype(
        {
            "frame": "int32",
            "time_seconds": "float32",
            "time_ms": "int32",
            "fps": "float32",

            "object_type": "string",
            "team": "string",

            "track_id": "Int32",

            "class_id": "int8",
            "class_name": "string",

            "confidence": "float32",

            "x_center": "float32",
            "y_center": "float32",

            "width": "float32",
            "height": "float32",

            "x_min": "float32",
            "y_min": "float32",
            "x_max": "float32",
            "y_max": "float32",

            "source_model": "string",
        }
    )

    print(
        f"Exporting {len(df)} tracking detections..."
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        "\nTracking parquet saved to:\n"
        f"{output_path}"
    )