from pathlib import Path
import shutil

prediction_dir = Path(
    "/data/manser/prediction_outputs/real_malaga_every_50_h1_ball_only_v12/predictions/labels"
)

manual_dir = Path(
    "/data/manser/datasets/ball_only/ball_only_dataset_v3/labels/train"
)

output_dir = Path(
    "/data/manser/datasets/ball_only/ball_only_dataset_v4/labels"
)

output_dir.mkdir(parents=True, exist_ok=True)

# 1. Alle Predictions kopieren
for file in prediction_dir.glob("*.txt"):
    shutil.copy2(file, output_dir / file.name)

# 2. Nur manuelle H1-Labels überschreiben
manual_h1_count = 0

for file in manual_dir.glob("h1_*.txt"):
    shutil.copy2(file, output_dir / file.name)
    manual_h1_count += 1

print(f"H1-Manuelle Labels übernommen: {manual_h1_count}")
print(f"Finale Labels: {len(list(output_dir.glob('*.txt')))}")