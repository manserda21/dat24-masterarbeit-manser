# =========================================================
# create_player_dataset.py
# =========================================================

import random
import shutil
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

# ORIGINAL CVAT YOLO EXPORT
SOURCE_DATASET_DIR = Path(
    "/data/manser/datasets/all_players_30_frames_dataset_v1"
)

# OUTPUT DATASET
OUTPUT_DATASET_DIR = Path(
    "/data/manser/datasets/real_players_dataset_v1_30_frames"
)

# SELECTED FULLY ANNOTATED FRAMES
SELECTED_FRAMES = [
    "frame_000000",
    "frame_000135",
    "frame_000315",
    "frame_000585",
    "frame_000930",
    "frame_001650",
    "frame_001965",
    "frame_002190",
    "frame_003210",
    "frame_004320",

    "frame_005100",
    "frame_005565",
    "frame_008250",
    "frame_009255",
    "frame_013440",
    "frame_014625",
    "frame_015120",
    "frame_015465",
    "frame_016155",
    "frame_016440",

    "frame_016785",
    "frame_018150",
    "frame_019140",
    "frame_025995",
    "frame_042510",
    "frame_044820",
    "frame_046605",
    "frame_058155",
    "frame_061500",
    "frame_061635",
]

# VALIDATION SPLIT
VAL_RATIO = 0.2

# IMAGE EXTENSION
IMAGE_EXT = ".png"

# PLAYER DATASET CLASSES ONLY
FINAL_CLASSES = {
    0: "modric",
    1: "ball",
    30: "kroos",
    31: "ronaldo",
    32: "benzema",
    33: "isco",
    34: "casemiro",
    35: "marcelo",
    36: "ramos",
    37: "varane",
    38: "danilo",
}

# REMAP TO NEW CLASS IDS
CLASS_ID_MAPPING = {
    0: 0,   # modric
    1: 1,   # ball
    30: 2,  # kroos
    31: 3,  # ronaldo
    32: 4,  # benzema
    33: 5,  # isco
    34: 6,  # casemiro
    35: 7,  # marcelo
    36: 8,  # ramos
    37: 9,  # varane
    38: 10, # danilo
}


# =========================================================
# RESET OUTPUT DIRECTORY
# =========================================================

if OUTPUT_DATASET_DIR.exists():
    shutil.rmtree(OUTPUT_DATASET_DIR)

# CREATE DIRECTORIES
for split in ["train", "val"]:

    (OUTPUT_DATASET_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT_DATASET_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True,
    )


# =========================================================
# TRAIN / VAL SPLIT
# =========================================================

random.seed(42)

frames = SELECTED_FRAMES.copy()
random.shuffle(frames)

val_size = int(len(frames) * VAL_RATIO)

val_frames = frames[:val_size]
train_frames = frames[val_size:]


# =========================================================
# HELPERS
# =========================================================

def process_label_file(input_path, output_path):

    filtered_lines = []

    with open(input_path, "r") as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_class_id = int(parts[0])

        if old_class_id not in CLASS_ID_MAPPING:
            continue

        new_class_id = CLASS_ID_MAPPING[old_class_id]

        parts[0] = str(new_class_id)

        filtered_lines.append(
            " ".join(parts)
        )

    with open(output_path, "w") as f:

        for line in filtered_lines:
            f.write(line + "\n")


def copy_split(split_name, split_frames):

    for frame_name in split_frames:

        image_name = frame_name + IMAGE_EXT
        label_name = frame_name + ".txt"

        source_image = (
            SOURCE_DATASET_DIR / "images" / "train" / image_name
        )

        source_label = (
            SOURCE_DATASET_DIR / "labels" / "train" / label_name
        )

        target_image = (
            OUTPUT_DATASET_DIR /
            "images" /
            split_name /
            image_name
        )

        target_label = (
            OUTPUT_DATASET_DIR /
            "labels" /
            split_name /
            label_name
        )

        if not source_image.exists():

            print(f"Missing image: {image_name}")
            continue

        if not source_label.exists():

            print(f"Missing label: {label_name}")
            continue

        shutil.copy2(
            source_image,
            target_image,
        )

        process_label_file(
            source_label,
            target_label,
        )


# =========================================================
# COPY DATA
# =========================================================

copy_split("train", train_frames)
copy_split("val", val_frames)


# =========================================================
# CREATE DATA.YAML
# =========================================================

yaml_content = f"""
path: {OUTPUT_DATASET_DIR}

train: images/train
val: images/val

names:
  0: modric
  1: ball
  2: kroos
  3: ronaldo
  4: benzema
  5: isco
  6: casemiro
  7: marcelo
  8: ramos
  9: varane
  10: danilo
"""

with open(
    OUTPUT_DATASET_DIR / "data.yaml",
    "w"
) as f:

    f.write(yaml_content.strip())


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("DATASET CREATED")
print("==============================")

print(f"\nOutput dataset:")
print(OUTPUT_DATASET_DIR)

print(f"\nTrain frames: {len(train_frames)}")
print(f"Validation frames: {len(val_frames)}")

print("\nTrain:")
for f in sorted(train_frames):
    print(f"  {f}")

print("\nValidation:")
for f in sorted(val_frames):
    print(f"  {f}")

print("\nClasses:")
for old_id, name in FINAL_CLASSES.items():

    new_id = CLASS_ID_MAPPING[old_id]

    print(f"  {new_id}: {name}")

print()