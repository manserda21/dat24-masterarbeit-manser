# =========================================================
# SCRIPT 3
# merge_manual_and_auto_labels.py
# =========================================================

from pathlib import Path
import shutil


# =========================================================
# CONFIG
# =========================================================

# ---------------------------------------------------------
# MANUAL DATASET
# ---------------------------------------------------------

MANUAL_IMAGE_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/selected/images/train"
)

MANUAL_LABEL_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/selected/labels/train"
)

# ---------------------------------------------------------
# AUTO PREDICTIONS
# ---------------------------------------------------------

AUTO_IMAGE_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/unlabeled_frames/images"
)

AUTO_LABEL_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/unlabeled_predictions/predict/labels"
)

# ---------------------------------------------------------
# FINAL DATASET
# ---------------------------------------------------------

FINAL_DATASET_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/final_cvat_dataset"
)

FINAL_IMAGE_DIR = FINAL_DATASET_DIR / "images"
FINAL_LABEL_DIR = FINAL_DATASET_DIR / "labels"


# =========================================================
# RESET OUTPUT
# =========================================================

if FINAL_DATASET_DIR.exists():
    shutil.rmtree(FINAL_DATASET_DIR)

FINAL_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# COPY MANUAL DATA
# =========================================================

manual_images = sorted(
    MANUAL_IMAGE_DIR.glob("*.png")
)

manual_labels = sorted(
    MANUAL_LABEL_DIR.glob("*.txt")
)


for image_path in manual_images:

    shutil.copy2(
        image_path,
        FINAL_IMAGE_DIR / image_path.name
    )


for label_path in manual_labels:

    shutil.copy2(
        label_path,
        FINAL_LABEL_DIR / label_path.name
    )


# =========================================================
# COPY AUTO DATA
# =========================================================

auto_images = sorted(
    list(AUTO_IMAGE_DIR.glob("*.png")) +
    list(AUTO_IMAGE_DIR.glob("*.jpg")) +
    list(AUTO_IMAGE_DIR.glob("*.jpeg"))
)

auto_labels = sorted(
    AUTO_LABEL_DIR.glob("*.txt")
)


for image_path in auto_images:

    shutil.copy2(
        image_path,
        FINAL_IMAGE_DIR / image_path.name
    )


for label_path in auto_labels:

    shutil.copy2(
        label_path,
        FINAL_LABEL_DIR / label_path.name
    )


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("DATASET MERGE FINISHED")
print("==============================")

print(f"Manual images: {len(manual_images)}")
print(f"Auto images:   {len(auto_images)}")

print(f"\nFinal dataset:")
print(FINAL_DATASET_DIR)

print("\nDone.\n")