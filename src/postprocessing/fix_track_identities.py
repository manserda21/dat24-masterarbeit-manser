# fix_track_identities.py

import argparse
from pathlib import Path

import pandas as pd


def calculate_track_purity(df):
    """
    Berechnet die Purity jeder Track-ID.
    """

    results = []

    for track_id, group in df.groupby("track_id"):

        counts = group["class_name"].value_counts()

        majority_class = counts.idxmax()
        majority_count = counts.max()

        total_count = len(group)

        purity = majority_count / total_count

        results.append(
            {
                "track_id": track_id,
                "majority_class": majority_class,
                "majority_count": majority_count,
                "total_count": total_count,
                "purity": purity,
                "num_classes": len(counts),
            }
        )

    return pd.DataFrame(results)


def smooth_track_identities(df, min_purity=0.7):
    """
    Ersetzt alle Klassen einer Track-ID durch die Mehrheitsklasse.
    """

    purity_df = calculate_track_purity(df)

    df_smoothed = df.copy()

    track_mapping = {}

    for _, row in purity_df.iterrows():

        track_id = row["track_id"]
        majority_class = row["majority_class"]
        purity = row["purity"]

        track_mapping[track_id] = {
            "majority_class": majority_class,
            "purity": purity,
            "accepted": purity >= min_purity,
        }

    df_smoothed["original_class_name"] = df_smoothed["class_name"]

    df_smoothed["track_purity"] = df_smoothed["track_id"].map(
        lambda x: track_mapping[x]["purity"]
    )

    df_smoothed["accepted_track"] = df_smoothed["track_id"].map(
        lambda x: track_mapping[x]["accepted"]
    )

    df_smoothed["class_name"] = df_smoothed["track_id"].map(
        lambda x: track_mapping[x]["majority_class"]
    )

    return df_smoothed, purity_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--purity_csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--min_purity",
        type=float,
        default=0.7,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    print(f"Loaded rows: {len(df):,}")

    df_smoothed, purity_df = smooth_track_identities(
        df,
        min_purity=args.min_purity,
    )

    Path(args.output_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_smoothed.to_csv(args.output_csv, index=False)
    purity_df.to_csv(args.purity_csv, index=False)

    print()
    print("========== RESULTS ==========")

    print(f"Tracks total: {len(purity_df)}")

    print(
        f"Accepted tracks (purity >= {args.min_purity:.2f}): "
        f"{(purity_df['purity'] >= args.min_purity).sum()}"
    )

    print(
        f"Rejected tracks: "
        f"{(purity_df['purity'] < args.min_purity).sum()}"
    )

    print(
        f"Mean purity: "
        f"{purity_df['purity'].mean():.3f}"
    )

    print()
    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.purity_csv}")


if __name__ == "__main__":
    main()