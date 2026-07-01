"""
Interaktives Track-Annotierungs-Tool
=====================================
Speichert für jeden Track ein Bild (3 Frames: Anfang/Mitte/Ende) in einen
Ordner. Du schaust das Bild an und gibst im Terminal den korrekten Namen ein.

Verwendung:
    python -m src.cleaning.annotate_tracks \
        --tracking  /data/.../cleaning_v2/tracking_clean.parquet \
        --summary   /data/.../cleaning_v2/track_summary.parquet \
        --video     /data/.../cleaning_v2/tracking_clean_visualization.mp4 \
        --output    /data/.../cleaning_v2/track_ground_truth.csv \
        --min_frames 30
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import pandas as pd


KNOWN_CLASSES = [
    "marcelo", "varane", "ramos", "danilo",
    "casemiro", "kroos", "modric", "isco",
    "benzema", "ronaldo", "opponent", "skip",
]


def save_track_image(video_path: str, tracking_df: pd.DataFrame, track_id: int,
                     dominant_class: str, track_length: int, purity: float,
                     num_team_changes: int, output_csv: str) -> str:
    """Speichert 3 Frames (Anfang, Mitte, Ende) des Tracks als PNG."""
    track_rows = tracking_df[tracking_df["track_id"] == track_id].sort_values("frame")
    if track_rows.empty:
        return None

    n = len(track_rows)
    cap = cv2.VideoCapture(video_path)
    panels = []

    for frac in [0.0, 0.5, 1.0]:
        idx = min(int(frac * (n - 1)), n - 1)
        row = track_rows.iloc[idx]
        frame_idx = int(row["frame"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1 = int(row["x_min"]), int(row["y_min"])
        x2, y2 = int(row["x_max"]), int(row["y_max"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"ID:{track_id} {row['class_name']}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        time_s = frame_idx / 25.0
        ts = f"F{frame_idx} | 10:{int(time_s)%60:02d}"
        cv2.putText(frame, ts, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        panels.append(cv2.resize(frame, (640, 360)))

    cap.release()

    if not panels:
        return None

    while len(panels) < 3:
        panels.append(panels[-1].copy())

    combined = cv2.hconcat(panels[:3])

    # Info-Banner oben
    banner = 50
    canvas = cv2.copyMakeBorder(combined, banner, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(30, 30, 30))
    info = (f"Track {track_id} | dominant: {dominant_class} | "
            f"purity: {purity:.2f} | length: {track_length} | "
            f"team_changes: {num_team_changes}")
    cv2.putText(canvas, info, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    img_dir = Path(output_csv).parent / "annotation_frames"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"track_{track_id:03d}_{dominant_class}.jpg"
    cv2.imwrite(str(img_path), canvas)
    return str(img_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking",   required=True)
    parser.add_argument("--summary",    required=True)
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--min_frames", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()

    tracking_df = pd.read_parquet(args.tracking)
    summary_df  = pd.read_parquet(args.summary)

    # Bereits annotierte Tracks laden
    output_path = Path(args.output)
    existing = {}
    if output_path.exists():
        with open(output_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[int(row["track_id"])] = row["correct_class"]
        print(f"  {len(existing)} bereits annotierte Tracks geladen.")

    relevant = summary_df[summary_df["track_length"] >= args.min_frames].sort_values(
        "track_length", ascending=False
    )
    todo = relevant[~relevant["track_id"].isin(existing.keys())]

    print(f"\n{len(relevant)} relevante Tracks, {len(todo)} noch zu annotieren.")
    print(f"Bekannte Klassen: {', '.join(KNOWN_CLASSES)}")
    print("Enter = dominant_class bestätigen | 'q' = speichern & beenden\n")
    print("─" * 60)

    results = dict(existing)

    for _, row in todo.iterrows():
        tid          = int(row["track_id"])
        dom          = row["dominant_class"]
        length       = int(row["track_length"])
        purity       = float(row["track_purity"])
        team_changes = int(row["num_team_changes"])

        print(f"\nTrack {tid:3d} | dominant: {dom:<12} | "
              f"purity: {purity:.2f} | length: {length:4d} | "
              f"team_changes: {team_changes}")

        img_path = save_track_image(
            args.video, tracking_df, tid, dom, length, purity,
            team_changes, args.output
        )
        if img_path:
            print(f"  Bild: {img_path}")

        while True:
            user_input = input(f"  Korrekte Klasse [{dom}]: ").strip().lower()

            if user_input == "":
                user_input = dom
                break
            elif user_input == "q":
                print("\nAnnotation unterbrochen. Bisherige Ergebnisse gespeichert.")
                _save(results, output_path)
                sys.exit(0)
            elif user_input in KNOWN_CLASSES:
                break
            else:
                print(f"  Unbekannt. Optionen: {', '.join(KNOWN_CLASSES)}")

        results[tid] = user_input
        print(f"  → gespeichert: {user_input}")
        _save(results, output_path)

    print(f"\nFertig! {len(results)} Tracks annotiert → {output_path}")


def _save(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "correct_class"])
        writer.writeheader()
        for tid, cls in sorted(results.items()):
            writer.writerow({"track_id": tid, "correct_class": cls})


if __name__ == "__main__":
    main()
