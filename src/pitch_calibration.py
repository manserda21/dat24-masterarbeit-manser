import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# SoccerPitch (reduzierte Version aus sn-calibration)
# ==================================================
class SoccerPitch:
    GOAL_LINE_TO_PENALTY_MARK = 11.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_HEIGHT = 2.44
    GOAL_LENGTH = 7.32

    def __init__(self, pitch_length=105.0, pitch_width=68.0):
        self.PITCH_LENGTH = pitch_length
        self.PITCH_WIDTH = pitch_width

        # Koordinatensystem:
        # x: links -> rechts
        # y: oben -> unten
        # Ursprung: linke obere Ecke = (0, 0)

        self.point_dict = {
            "TL_PITCH_CORNER": np.array([0.0, 0.0, 0.0]),
            "BL_PITCH_CORNER": np.array([0.0, pitch_width, 0.0]),
            "TR_PITCH_CORNER": np.array([pitch_length, 0.0, 0.0]),
            "BR_PITCH_CORNER": np.array([pitch_length, pitch_width, 0.0]),

            "L_PENALTY_AREA_TL_CORNER": np.array([0.0, (pitch_width - self.PENALTY_AREA_WIDTH) / 2.0, 0.0]),
            "L_PENALTY_AREA_TR_CORNER": np.array([self.PENALTY_AREA_LENGTH, (pitch_width - self.PENALTY_AREA_WIDTH) / 2.0, 0.0]),
            "L_PENALTY_AREA_BL_CORNER": np.array([0.0, (pitch_width + self.PENALTY_AREA_WIDTH) / 2.0, 0.0]),
            "L_PENALTY_AREA_BR_CORNER": np.array([self.PENALTY_AREA_LENGTH, (pitch_width + self.PENALTY_AREA_WIDTH) / 2.0, 0.0]),

            "L_GOAL_AREA_TL_CORNER": np.array([0.0, (pitch_width - self.GOAL_AREA_WIDTH) / 2.0, 0.0]),
            "L_GOAL_AREA_TR_CORNER": np.array([self.GOAL_AREA_LENGTH, (pitch_width - self.GOAL_AREA_WIDTH) / 2.0, 0.0]),
            "L_GOAL_AREA_BL_CORNER": np.array([0.0, (pitch_width + self.GOAL_AREA_WIDTH) / 2.0, 0.0]),
            "L_GOAL_AREA_BR_CORNER": np.array([self.GOAL_AREA_LENGTH, (pitch_width + self.GOAL_AREA_WIDTH) / 2.0, 0.0]),
        }

        self.line_extremities = {
            "Side line top": (
                self.point_dict["TL_PITCH_CORNER"],
                self.point_dict["TR_PITCH_CORNER"],
            ),
            "Side line left": (
                self.point_dict["TL_PITCH_CORNER"],
                self.point_dict["BL_PITCH_CORNER"],
            ),
            "Big rect. left top": (
                self.point_dict["L_PENALTY_AREA_TL_CORNER"],
                self.point_dict["L_PENALTY_AREA_TR_CORNER"],
            ),
            "Small rect. left top": (
                self.point_dict["L_GOAL_AREA_TL_CORNER"],
                self.point_dict["L_GOAL_AREA_TR_CORNER"],
            ),
            "Small rect. left bottom": (
                self.point_dict["L_GOAL_AREA_BL_CORNER"],
                self.point_dict["L_GOAL_AREA_BR_CORNER"],
            ),
            "Small rect. left main": (
                self.point_dict["L_GOAL_AREA_TR_CORNER"],
                self.point_dict["L_GOAL_AREA_BR_CORNER"],
            ),
        }

    def get_2d_homogeneous_line(self, line_name):
        if line_name not in self.line_extremities:
            return None

        extremities = self.line_extremities[line_name]
        p1 = np.array([extremities[0][0], extremities[0][1], 1.0], dtype=np.float64)
        p2 = np.array([extremities[1][0], extremities[1][1], 1.0], dtype=np.float64)

        line = np.cross(p1, p2)
        n = np.linalg.norm(line[:2])
        if n > 0:
            line = line / n
        return line


# ==================================================
# Dateien laden
# ==================================================
base_path = Path("/data/manser/soccernet/calibration-2023/train")
image_path = base_path / "16412.jpg"
json_path = base_path / "16412.json"

img_bgr = cv2.imread(str(image_path))
if img_bgr is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_h, img_w = img_bgr.shape[:2]

with open(json_path, "r") as f:
    annotations = json.load(f)


# ==================================================
# Hilfsfunktionen
# ==================================================
def norm_to_pixel(p, width, height):
    return np.array([p["x"] * width, p["y"] * height], dtype=np.float64)


def get_segment(name, ann, width, height):
    pts = ann[name]
    if len(pts) != 2:
        raise ValueError(f"{name} hat nicht genau 2 Punkte.")
    p1 = norm_to_pixel(pts[0], width, height)
    p2 = norm_to_pixel(pts[1], width, height)
    return p1, p2


def line_from_points(p1, p2):
    hp1 = np.array([p1[0], p1[1], 1.0], dtype=np.float64)
    hp2 = np.array([p2[0], p2[1], 1.0], dtype=np.float64)
    l = np.cross(hp1, hp2)
    n = np.linalg.norm(l[:2])
    if n > 0:
        l = l / n
    return l


def normalization_transform(points):
    points = np.asarray(points, dtype=np.float64)
    center = np.mean(points, axis=0)

    d = 0.0
    nelems = 0
    for p in points:
        nelems += 1
        di = np.linalg.norm(p - center)
        d += (di - d) / nelems

    s = 1.0 if d <= 0 else np.sqrt(2) / d

    T = np.zeros((3, 3), dtype=np.float64)
    T[0, 0] = s
    T[0, 2] = -s * center[0]
    T[1, 1] = s
    T[1, 2] = -s * center[1]
    T[2, 2] = 1.0
    return T


def estimate_homography_from_line_correspondences(lines, T1=np.eye(3), T2=np.eye(3)):
    H = np.eye(3, dtype=np.float64)
    A = np.zeros((len(lines) * 2, 9), dtype=np.float64)

    for i, (src_line_raw, target_line_raw) in enumerate(lines):
        src_line = np.transpose(np.linalg.inv(T1)) @ src_line_raw
        target_line = np.transpose(np.linalg.inv(T2)) @ target_line_raw

        u, v, w = src_line
        x, y, z = target_line

        A[2 * i, 0] = 0
        A[2 * i, 1] = x * w
        A[2 * i, 2] = -x * v
        A[2 * i, 3] = 0
        A[2 * i, 4] = y * w
        A[2 * i, 5] = -v * y
        A[2 * i, 6] = 0
        A[2 * i, 7] = z * w
        A[2 * i, 8] = -v * z

        A[2 * i + 1, 0] = x * w
        A[2 * i + 1, 1] = 0
        A[2 * i + 1, 2] = -x * u
        A[2 * i + 1, 3] = y * w
        A[2 * i + 1, 4] = 0
        A[2 * i + 1, 5] = -u * y
        A[2 * i + 1, 6] = z * w
        A[2 * i + 1, 7] = 0
        A[2 * i + 1, 8] = -u * z

    _, s, vh = np.linalg.svd(A)
    if s[-1] <= 0:
        return False, H

    v = np.reshape(vh[-1], (3, 3))
    H = np.linalg.inv(T2) @ v @ T1

    if abs(H[2, 2]) > 1e-12:
        H /= H[2, 2]

    return True, H


def draw_segment(img, p1, p2, color=(0, 255, 0), thickness=2):
    p1i = tuple(np.round(p1).astype(int))
    p2i = tuple(np.round(p2).astype(int))
    cv2.line(img, p1i, p2i, color, thickness)


def transform_point(H, p):
    hp = np.array([p[0], p[1], 1.0], dtype=np.float64)
    q = H @ hp
    q /= q[2]
    return np.array([q[0], q[1]], dtype=np.float64)


# ==================================================
# Liniennamen wie im SoccerNet-Repo
# ==================================================
used_line_names = [
    "Side line left",
    "Side line top",
    "Big rect. left top",
    "Small rect. left top",
    "Small rect. left bottom",
    "Small rect. left main",
]

# Bildlinien
image_segments = {}
for name in used_line_names:
    if name not in annotations:
        raise KeyError(f"{name} fehlt in {json_path.name}")
    image_segments[name] = get_segment(name, annotations, img_w, img_h)

# Pitch-Linien direkt aus SoccerPitch
pitch = SoccerPitch(pitch_length=105.0, pitch_width=68.0)

line_matches = []
src_pts = []
target_pts = []

for name in used_line_names:
    pitch_line = pitch.get_2d_homogeneous_line(name)

    ip1, ip2 = image_segments[name]
    image_line = line_from_points(ip1, ip2)

    line_matches.append((pitch_line, image_line))

    pp1_3d, pp2_3d = pitch.line_extremities[name]
    pp1 = np.array([pp1_3d[0], pp1_3d[1]], dtype=np.float64)
    pp2 = np.array([pp2_3d[0], pp2_3d[1]], dtype=np.float64)

    src_pts.extend([pp1, pp2])
    target_pts.extend([ip1, ip2])

src_pts = np.asarray(src_pts, dtype=np.float64)
target_pts = np.asarray(target_pts, dtype=np.float64)

T_pitch = normalization_transform(src_pts)
T_img = normalization_transform(target_pts)

success, H_pitch_to_image = estimate_homography_from_line_correspondences(
    line_matches, T1=T_pitch, T2=T_img
)

if not success:
    raise RuntimeError("Homography konnte nicht geschaetzt werden.")

print("Homography (Pitch -> Image):")
print(H_pitch_to_image)

H_image_to_pitch = np.linalg.inv(H_pitch_to_image)
H_image_to_pitch /= H_image_to_pitch[2, 2]


# ==================================================
# Visualisierung der verwendeten Linien
# ==================================================
img_vis = img_rgb.copy()

for idx, name in enumerate(used_line_names):
    p1, p2 = image_segments[name]
    draw_segment(img_vis, p1, p2, color=(0, 255, 0), thickness=3)

    mid = (p1 + p2) / 2.0
    cv2.putText(
        img_vis,
        f"{idx}:{name}",
        tuple(np.round(mid).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

plt.figure(figsize=(14, 8))
plt.imshow(img_vis)
plt.title("Verwendete Linien aus 16412.jpg")
plt.axis("off")
plt.show()


# ==================================================
# Bird's-Eye-View
# ==================================================
scale = 10
pitch_w_px = int(105 * scale)
pitch_h_px = int(68 * scale)

S = np.array([
    [scale, 0, 0],
    [0, scale, 0],
    [0, 0, 1]
], dtype=np.float64)

H_image_to_pitch_scaled = S @ H_image_to_pitch

bird = cv2.warpPerspective(
    img_rgb,
    H_image_to_pitch_scaled,
    (pitch_w_px, pitch_h_px)
)

plt.figure(figsize=(14, 8))
plt.imshow(bird)
plt.title("Bird's-Eye-View mit SoccerPitch")
plt.axis("off")
plt.show()


# ==================================================
# Transformierte Linien auf leerem Pitch anzeigen
# ==================================================
pitch_canvas = np.ones((pitch_h_px, pitch_w_px, 3), dtype=np.uint8) * 255
cv2.rectangle(pitch_canvas, (0, 0), (pitch_w_px - 1, pitch_h_px - 1), (0, 0, 0), 2)

for name in used_line_names:
    p1, p2 = image_segments[name]
    q1 = transform_point(H_image_to_pitch_scaled, p1)
    q2 = transform_point(H_image_to_pitch_scaled, p2)
    draw_segment(pitch_canvas, q1, q2, color=(255, 0, 0), thickness=2)

plt.figure(figsize=(14, 8))
plt.imshow(pitch_canvas)
plt.title("Transformierte Linien auf dem 105x68-Pitch")
plt.axis("off")
plt.show()