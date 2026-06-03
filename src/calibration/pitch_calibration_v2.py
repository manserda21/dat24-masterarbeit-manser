import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2


# =========================================
# CONFIG
# =========================================

DATASET_PATH = "/data/manser/datasets/calibration/malaga_real_2016_pitch_calibration_dataset_points_lines_v1"
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations.xml")
IMAGES_PATH = os.path.join(DATASET_PATH, "images")

OUTPUT_DIR = "/data/manser/homography_outputs/malaga_real_calib_v2"

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
SCALE = 10  # Pixel pro Meter


# =========================================
# WORLD COORDINATES
# =========================================

WORLD_POINTS = {
    "TL_PITCH_CORNER": (0, 0),
    "TR_PITCH_CORNER": (PITCH_WIDTH, 0),
    "BL_PITCH_CORNER": (0, PITCH_HEIGHT),
    "BR_PITCH_CORNER": (PITCH_WIDTH, PITCH_HEIGHT),

    "CENTER_POINT": (PITCH_WIDTH / 2, PITCH_HEIGHT / 2),

    "PENALTY_SPOT_LEFT": (11, PITCH_HEIGHT / 2),
    "PENALTY_SPOT_RIGHT": (PITCH_WIDTH - 11, PITCH_HEIGHT / 2),

    "CENTER_CIRCLE_TOP": (PITCH_WIDTH / 2, (PITCH_HEIGHT / 2) - 9.15),
    "CENTER_CIRCLE_BOTTOM": (PITCH_WIDTH / 2, (PITCH_HEIGHT / 2) + 9.15),
    "CENTER_CIRCLE_LEFT": ((PITCH_WIDTH / 2) - 9.15, PITCH_HEIGHT / 2),
    "CENTER_CIRCLE_RIGHT": ((PITCH_WIDTH / 2) + 9.15, PITCH_HEIGHT / 2),
}


# =========================================
# XML PARSING
# =========================================

def load_annotations(path):
    tree = ET.parse(path)
    root = tree.getroot()

    frames = {}

    for image in root.findall("image"):
        name = image.get("name")

        points = []
        lines = {}

        for p in image.findall("points"):
            label = p.get("label")
            x, y = map(float, p.get("points").split(","))
            points.append((label, (x, y)))

        for pl in image.findall("polyline"):
            label = pl.get("label")
            coords = []

            for pair in pl.get("points").split(";"):
                x, y = map(float, pair.split(","))
                coords.append((x, y))

            lines[label] = coords

        frames[name] = {
            "points": points,
            "lines": lines
        }

    return frames


# =========================================
# BUILD CORRESPONDENCES
# =========================================

def build_correspondences(frame):
    img_pts = []
    world_pts = []

    IMPORTANT_POINTS = [
        "CENTER_POINT",
        "CENTER_CIRCLE_TOP",
        "CENTER_CIRCLE_BOTTOM",
        "CENTER_CIRCLE_LEFT",
        "CENTER_CIRCLE_RIGHT",
        "PENALTY_SPOT_LEFT",
        "PENALTY_SPOT_RIGHT"
    ]

    for label, coord in frame["points"]:
        if label in IMPORTANT_POINTS:
            img_pts.append(coord)
            world_pts.append(WORLD_POINTS[label])

    return np.array(img_pts), np.array(world_pts)


# =========================================
# HOMOGRAPHY
# =========================================

def compute_homography(frame):
    img_pts, world_pts = build_correspondences(frame)

    if len(img_pts) < 4:
        return None

    world_pts = world_pts.astype(np.float32) * SCALE

    H, mask = cv2.findHomography(
        img_pts,
        world_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    return H


# =========================================
# WARP
# =========================================

def warp(image_path, H):
    img = cv2.imread(image_path)

    width = int(PITCH_WIDTH * SCALE)
    height = int(PITCH_HEIGHT * SCALE)

    warped = cv2.warpPerspective(img, H, (width, height))

    return img, warped


# =========================================
# MAIN (BATCH!)
# =========================================

def main():
    frames = load_annotations(ANNOTATIONS_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    homographies = {}

    for name, frame in frames.items():

        print(f"\nProcessing: {name}")

        H = compute_homography(frame)

        if H is None:
            print(" -> skipped (not enough valid points)")
            continue

        img_path = os.path.join(IMAGES_PATH, name)

        if not os.path.exists(img_path):
            print(" -> image not found")
            continue

        img, warped = warp(img_path, H)

        # speichern
        original_path = os.path.join(OUTPUT_DIR, f"{name}_original.png")
        bev_path = os.path.join(OUTPUT_DIR, f"{name}_bev.png")

        cv2.imwrite(original_path, img)
        cv2.imwrite(bev_path, warped)

        homographies[name] = H

        print(" -> saved")

    # =========================================
    # SAVE HOMOGRAPHIES
    # =========================================

    np.savez(
        os.path.join(OUTPUT_DIR, "homographies.npz"),
        **homographies
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()