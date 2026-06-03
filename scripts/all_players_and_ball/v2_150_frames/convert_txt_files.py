from pathlib import Path

# =========================================================
# INPUT
# =========================================================

INPUT_TXT = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/train.txt"
)

# =========================================================
# OUTPUT
# =========================================================

OUTPUT_TXT = Path(
    "/data/manser/datasets/all_players_and_ball/final_cvat_dataset/train.txt"
)

# =========================================================
# PROCESS
# =========================================================

with open(INPUT_TXT, "r") as f:
    lines = f.readlines()

relative_paths = []

for line in lines:

    line = line.strip()

    if not line:
        continue

    # Nur Dateiname extrahieren
    frame_name = Path(line).name

    # CVAT-kompatibler Pfad
    relative_path = f"images/train/{frame_name}"

    relative_paths.append(relative_path)

# =========================================================
# SAVE
# =========================================================

OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_TXT, "w") as f:

    for relative_path in relative_paths:
        f.write(relative_path + "\n")

# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("TRAIN.TXT CREATED")
print("==============================")

print(f"Input file:  {INPUT_TXT}")
print(f"Output file: {OUTPUT_TXT}")

print(f"\nImages listed: {len(relative_paths)}")

print("\nFirst 10 entries:\n")

for path in relative_paths[:10]:
    print(path)

print("\nDone.\n")