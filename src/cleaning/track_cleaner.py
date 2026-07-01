# src/cleaning/track_cleaner.py

import pandas as pd


class TrackCleaner:
    def __init__(
        self,
        min_track_length: int = 10,
        review_purity_threshold: float = 0.8,
        review_length_threshold: int = 50,
    ):
        self.min_track_length = min_track_length
        self.review_purity_threshold = review_purity_threshold
        self.review_length_threshold = review_length_threshold

    def clean(
        self,
        tracking_df: pd.DataFrame,
    ):
        player_df = tracking_df[
            tracking_df["object_type"] == "player"
        ].copy()

        ball_df = tracking_df[
            tracking_df["object_type"] == "ball"
        ].copy()

        track_summaries = []

        for track_id, group in player_df.groupby("track_id"):

            group = group.sort_values("frame")

            track_length = len(group)

            class_counts = group["class_name"].value_counts()
            dominant_class = class_counts.idxmax()
            dominant_count = class_counts.max()
            purity = dominant_count / track_length

            team_sequence = group["team"].tolist()

            num_team_changes = 0

            for previous_team, current_team in zip(
                team_sequence,
                team_sequence[1:],
            ):
                if previous_team != current_team:
                    num_team_changes += 1

            num_classes = group["class_name"].nunique()
            num_teams = group["team"].nunique()
            review_flag = (
                track_length
                >= self.review_length_threshold
                and purity
                < self.review_purity_threshold
            )

            track_summaries.append(
                {
                    "track_id": track_id,
                    "track_length": track_length,
                    "dominant_class": dominant_class,
                    "track_purity": purity,
                    "num_classes": num_classes,
                    "num_teams": num_teams,
                    "num_team_changes": num_team_changes,
                    "review_flag": review_flag,
                }
            )

        summary_df = pd.DataFrame(track_summaries)

        player_df = player_df.merge(
            summary_df,
            on="track_id",
            how="left",
        )

        # Changed:
        # Do not overwrite frame-level classifications in V1.
        # Majority vote is only stored as dominant_class.
        player_df["clean_class_name"] = player_df["class_name"]

        player_df = player_df[
            player_df["track_length"] >= self.min_track_length
        ]

        ball_df["track_length"] = None
        ball_df["dominant_class"] = "ball"
        ball_df["track_purity"] = 1.0
        ball_df["num_classes"] = 1
        ball_df["num_teams"] = 1
        ball_df["num_team_changes"] = 0
        ball_df["review_flag"] = False
        ball_df["clean_class_name"] = "ball"

        cleaned_df = pd.concat(
            [
                player_df,
                ball_df,
            ],
            ignore_index=True,
        )

        cleaned_df = cleaned_df.sort_values(
            [
                "frame",
                "object_type",
            ]
        )

        return cleaned_df, summary_df