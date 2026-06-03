from pathlib import Path
import shutil


# =========================================================
# CONFIG
# =========================================================

SOURCE_DATASET_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames"
)

OUTPUT_DATASET_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/selected"
)

IMAGE_DIR = SOURCE_DATASET_DIR / "images" / "train" / "datasets" / "real_frames_every_50"
LABEL_DIR = SOURCE_DATASET_DIR / "labels" / "train" / "datasets" / "real_frames_every_50"

OUTPUT_IMAGE_DIR = OUTPUT_DATASET_DIR / "images" / "train"
OUTPUT_LABEL_DIR = OUTPUT_DATASET_DIR / "labels" / "train"


# =========================================================
# SELECTED FRAMES
# =========================================================

SELECTED_FRAMES = [
    "frame_000000",
    "frame_000050",
    "frame_000100",
    "frame_000150",
    "frame_000200",
    "frame_000250",
    "frame_000300",
    "frame_000550",
    "frame_000600",
    "frame_000800",
    "frame_000850",
    "frame_001000",
    "frame_001150",
    "frame_001200",
    "frame_001250",
    "frame_001300",
    "frame_001350",
    "frame_001400",
    "frame_001500",
    "frame_001550",
    "frame_001600",
    "frame_001700",
    "frame_001800",
    "frame_002050",
    "frame_002200",
    "frame_002300",
    "frame_002450",
    "frame_002500",
    "frame_002600",
    "frame_002750",
    "frame_003200",
    "frame_003250",
    "frame_003550",
    "frame_003600",
    "frame_004250",
    "frame_004400",
    "frame_004900",
    "frame_005050",
    "frame_005250",
    "frame_005500",
    "frame_005750",
    "frame_005950",
    "frame_006150",
    "frame_006250",
    "frame_007100",
    "frame_007150",
    "frame_007200",
    "frame_007600",
    "frame_008250",
    "frame_008350",
    "frame_009000",
    "frame_009250",
    "frame_009400",
    "frame_009550",
    "frame_009750",
    "frame_010000",
    "frame_010250",
    "frame_011600",
    "frame_012000",
    "frame_012400",
    "frame_012700",
    "frame_012850",
    "frame_013150",
    "frame_013300",
    "frame_014900",
    "frame_015300",
    "frame_015450",
    "frame_015600",
    "frame_015900",
    "frame_016350",
    "frame_016600",
    "frame_016850",
    "frame_017250",
    "frame_017400",
    "frame_017500",
    "frame_017750",
    "frame_017850",
    "frame_018150",
    "frame_018300",
    "frame_018500",
    "frame_018800",
    "frame_019100",
    "frame_019300",
    "frame_022350",
    "frame_024000",
    "frame_024500",
    "frame_024800",
    "frame_026050",
    "frame_026450",
    "frame_026900",
    "frame_027200",
    "frame_027650",
    "frame_028150",
    "frame_028850",
    "frame_029400",
    "frame_029750",
    "frame_031000",
    "frame_032500",
    "frame_033100",
    "frame_033350",
    "frame_033500",
    "frame_037500",
    "frame_038000",
    "frame_042000",
    "frame_042450",
    "frame_042500",
    "frame_043500",
    "frame_045650",
    "frame_048200",
    "frame_049000",
    "frame_050000",
    "frame_051150",
    "frame_052000",
    "frame_052650",
]


# =========================================================
# RESET OUTPUT DIRECTORY
# =========================================================

if OUTPUT_DATASET_DIR.exists():
    shutil.rmtree(OUTPUT_DATASET_DIR)

OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# COPY FILES
# =========================================================

copied_images = 0
copied_labels = 0
missing_images = []
missing_labels = []


for frame_name in SELECTED_FRAMES:

    image_path = IMAGE_DIR / f"{frame_name}.png"
    label_path = LABEL_DIR / f"{frame_name}.txt"

    output_image_path = OUTPUT_IMAGE_DIR / image_path.name
    output_label_path = OUTPUT_LABEL_DIR / label_path.name

    # -----------------------------------------------------
    # COPY IMAGE
    # -----------------------------------------------------

    if image_path.exists():

        shutil.copy2(
            image_path,
            output_image_path
        )

        copied_images += 1

    else:
        missing_images.append(image_path.name)

    # -----------------------------------------------------
    # COPY LABEL
    # -----------------------------------------------------

    if label_path.exists():

        shutil.copy2(
            label_path,
            output_label_path
        )

        copied_labels += 1

    else:
        missing_labels.append(label_path.name)


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("DATASET EXTRACTION FINISHED")
print("==============================")

print(f"Selected frames: {len(SELECTED_FRAMES)}")
print(f"Copied images:   {copied_images}")
print(f"Copied labels:   {copied_labels}")

print(f"\nOutput dataset:")
print(OUTPUT_DATASET_DIR)

print("\nMissing images:")
print(len(missing_images))

print("Missing labels:")
print(len(missing_labels))

print("\nDone.\n")