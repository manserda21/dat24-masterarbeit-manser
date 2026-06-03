from pathlib import Path
import shutil
import random


# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.8

SOURCE_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v3_175_frames/selected"
)

OUTPUT_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v3_175_frames/selected_split"
)

SOURCE_IMAGE_DIR = SOURCE_DIR / "images" / "train"
SOURCE_LABEL_DIR = SOURCE_DIR / "labels" / "train"


# =========================================================
# RESET OUTPUT DIRECTORY
# =========================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# COLLECT FRAMES
# =========================================================

image_paths = sorted(
    SOURCE_IMAGE_DIR.glob("*.png")
)

samples = []

for image_path in image_paths:

    label_path = (
        SOURCE_LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        continue

    samples.append(
        (image_path, label_path)
    )


# =========================================================
# TRAIN / VAL SPLIT
# =========================================================

random.seed(RANDOM_SEED)

random.shuffle(samples)

n_train = int(len(samples) * TRAIN_RATIO)

train_samples = samples[:n_train]
val_samples = samples[n_train:]


# =========================================================
# COPY FILES
# =========================================================

def copy_samples(samples, split):

    for image_path, label_path in samples:

        shutil.copy2(
            image_path,
            OUTPUT_DIR
            / "images"
            / split
            / image_path.name
        )

        shutil.copy2(
            label_path,
            OUTPUT_DIR
            / "labels"
            / split
            / label_path.name
        )


copy_samples(train_samples, "train")

copy_samples(val_samples, "val")


# =========================================================
# CREATE DATA.YAML
# =========================================================

yaml_content = """
path: /data/manser/datasets/all_players_and_ball/all_players_and_ball_v3_175_frames/selected_split

train: images/train
val: images/val

names:
  0: opponent
  1: danilo
  2: varane
  3: ramos
  4: marcelo
  5: modric
  6: casemiro
  7: kroos
  8: isco
  9: benzema
  10: ronaldo
  11: referee
  12: person_temp
  13: ball
"""

(
    OUTPUT_DIR / "data.yaml"
).write_text(yaml_content.strip())


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("DATASET SPLIT FINISHED")
print("==============================")

print(f"Total samples: {len(samples)}")
print(f"Train samples: {len(train_samples)}")
print(f"Val samples:   {len(val_samples)}")

print("\nOutput:")
print(OUTPUT_DIR)

print("\nDone.\n")