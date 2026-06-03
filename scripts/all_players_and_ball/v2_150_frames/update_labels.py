# THe labels for the predictions contain the confidence score, which is not needed for the evaluation.
# This script removes the confidence score from the labels.

from pathlib import Path

# Pfad zu den Label-Dateien
LABEL_DIR = Path("/data/manser/datasets/all_players_and_ball/final_cvat_dataset/labels/train")

# Alle TXT-Dateien durchgehen
for txt_file in LABEL_DIR.glob("*.txt"):

    cleaned_lines = []

    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        # Falls 6 Werte vorhanden sind:
        # class x y w h confidence
        # -> confidence entfernen
        if len(parts) == 6:
            parts = parts[:5]

        cleaned_lines.append(" ".join(parts))

    # Datei überschreiben
    with open(txt_file, "w") as f:
        f.write("\n".join(cleaned_lines))

print("Alle Label-Dateien wurden bereinigt.")