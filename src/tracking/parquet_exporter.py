# src/tracking/parquet_exporter.py

from pathlib import Path

import pandas as pd


TRACKING_COLUMNS = [
    "frame",
    "time_seconds",
    "time_ms",
    "fps",
    "object_type",
    "team",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x_center",
    "y_center",
    "width",
    "height",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "source_model",
]


DTYPES = {
    "frame": "int32",
    "time_seconds": "float32",
    "time_ms": "int32",
    "fps": "float32",
    "object_type": "string",
    "team": "string",
    "track_id": "Int32",
    "class_id": "Int16",
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


def export_rows_to_parquet(
    rows: list[dict],
    output_path: str,
) -> None:
    """
    Export a list of tracking rows to parquet.

    The rows must already follow the common tracking schema.
    """

    if not rows:
        print("No tracking rows found.")
        return

    df = pd.DataFrame(rows)

    for column in TRACKING_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[TRACKING_COLUMNS]

    df = df.astype(DTYPES)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Exporting {len(df)} tracking rows...")

    df.to_parquet(
        output_path,
        index=False,
    )

    print(
        "\nTracking parquet saved to:\n"
        f"{output_path}"
    )