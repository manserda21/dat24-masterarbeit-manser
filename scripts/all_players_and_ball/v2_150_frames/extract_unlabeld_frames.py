# =========================================================
# SCRIPT 1
# extract_unlabeled_frames.py
# =========================================================

from pathlib import Path
import shutil


# =========================================================
# CONFIG
# =========================================================

SOURCE_IMAGE_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/all_players_and_ball_v2_150_frames/images/train/datasets/real_frames_every_50"
)

OUTPUT_DIR = Path(
    "/data/manser/datasets/all_players_and_ball/unlabeled_frames"
)

OUTPUT_IMAGE_DIR = OUTPUT_DIR / "images"


# =========================================================
# ALREADY ANNOTATED FRAMES
# =========================================================

ANNOTATED_FRAMES = {

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
}


# =========================================================
# RESET OUTPUT
# =========================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# EXTRACT UNLABELED FRAMES
# =========================================================

all_images = sorted(
    SOURCE_IMAGE_DIR.glob("*.png")
)

copied = 0


for image_path in all_images:

    frame_name = image_path.stem

    if frame_name not in ANNOTATED_FRAMES:

        shutil.copy2(
            image_path,
            OUTPUT_IMAGE_DIR / image_path.name
        )

        copied += 1


# =========================================================
# SUMMARY
# =========================================================

print("\n==============================")
print("UNLABELED FRAME EXTRACTION")
print("==============================")

print(f"Total images:      {len(all_images)}")
print(f"Annotated images:  {len(ANNOTATED_FRAMES)}")
print(f"Unlabeled images:  {copied}")

print(f"\nOutput:")
print(OUTPUT_DIR)

print("\nDone.\n")