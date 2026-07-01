import argparse
from pathlib import Path

import pandas as pd

from src.cleaning.track_cleaner import (
    TrackCleaner,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Tracking parquet",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print(
        "Loading tracking parquet..."
    )

    tracking_df = pd.read_parquet(
        args.input
    )

    cleaner = TrackCleaner()

    (
        cleaned_df,
        summary_df,
    ) = cleaner.clean(
        tracking_df
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned_path = (
        output_dir
        / "tracking_clean.parquet"
    )

    summary_path = (
        output_dir
        / "track_summary.parquet"
    )

    cleaned_df.to_parquet(
        cleaned_path,
        index=False,
    )

    summary_df.to_parquet(
        summary_path,
        index=False,
    )

    print(
        f"\nCleaned tracking saved:\n"
        f"{cleaned_path}"
    )

    print(
        f"\nTrack summary saved:\n"
        f"{summary_path}"
    )

    print("\nStatistics")

    print(
        f"Tracks: {len(summary_df)}"
    )

    print(
        f"Review tracks: "
        f"{summary_df['review_flag'].sum()}"
    )

    print(
        f"Mean purity: "
        f"{summary_df['track_purity'].mean():.3f}"
    )


if __name__ == "__main__":
    main()