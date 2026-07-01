# calibration/pitch_calibration_v4.py
#
# Improvements over soccernet_homography_dataset_v2.py:
#   1. Near-parallel line pair filtering (angle < min_line_angle_deg are skipped).
#   2. Non-linear Levenberg-Marquardt refinement after RANSAC (all inliers used).
#   3. Symmetric reprojection error (image -> field AND field -> image) for quality.
#   4. Per-frame debug overlays: full pitch model projected back into the image.
#   5. Saves all homographies as a single .npz in addition to per-frame .txt files.
#
# Usage:
#   python pitch_calibration_v4.py \
#     --dataset /data/manser/datasets/calibration/malaga_home_stadium_v1 \
#     --output  /data/manser/homography_outputs/malaga_v4 \
#     --debug-overlays

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Pitch geometry (metres, FIFA standard)
# ---------------------------------------------------------------------------

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PENALTY_DEPTH = 16.5
PENALTY_Y_TOP = (PITCH_WIDTH - 40.32) / 2.0
PENALTY_Y_BOTTOM = (PITCH_WIDTH + 40.32) / 2.0
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_Y_TOP = (PITCH_WIDTH - 18.32) / 2.0
GOAL_AREA_Y_BOTTOM = (PITCH_WIDTH + 18.32) / 2.0
CENTER_X = PITCH_LENGTH / 2.0
CENTER_Y = PITCH_WIDTH / 2.0
CENTER_CIRCLE_RADIUS = 9.15

# ---------------------------------------------------------------------------
# World line model:  ax + by + c = 0  (unit-normalised by construction)
# ---------------------------------------------------------------------------

WORLD_LINES: Dict[str, Tuple[float, float, float]] = {
    "Side line top":          (0.0, 1.0, 0.0),
    "Side line bottom":       (0.0, 1.0, -PITCH_WIDTH),
    "Side line left":         (1.0, 0.0, 0.0),
    "Side line right":        (1.0, 0.0, -PITCH_LENGTH),
    "Middle line":            (1.0, 0.0, -CENTER_X),
    "Big rect. left main":    (1.0, 0.0, -PENALTY_DEPTH),
    "Big rect. left top":     (0.0, 1.0, -PENALTY_Y_TOP),
    "Big rect. left bottom":  (0.0, 1.0, -PENALTY_Y_BOTTOM),
    "Big rect. right main":   (1.0, 0.0, -(PITCH_LENGTH - PENALTY_DEPTH)),
    "Big rect. right top":    (0.0, 1.0, -PENALTY_Y_TOP),
    "Big rect. right bottom": (0.0, 1.0, -PENALTY_Y_BOTTOM),
    "Small rect. left main":  (1.0, 0.0, -GOAL_AREA_DEPTH),
    "Small rect. left top":   (0.0, 1.0, -GOAL_AREA_Y_TOP),
    "Small rect. left bottom":(0.0, 1.0, -GOAL_AREA_Y_BOTTOM),
    "Small rect. right main": (1.0, 0.0, -(PITCH_LENGTH - GOAL_AREA_DEPTH)),
    "Small rect. right top":  (0.0, 1.0, -GOAL_AREA_Y_TOP),
    "Small rect. right bottom":(0.0, 1.0, -GOAL_AREA_Y_BOTTOM),
}

# Circle labels that exist in SoccerNet but we handle separately.
CIRCLE_LABELS = {"Circle central", "Circle left", "Circle right"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_image_path(images_dir: Path, frame_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fit_image_line(points_px: np.ndarray) -> Optional[np.ndarray]:
    """Fit normalised homogeneous line ax+by+c=0 via total least squares."""
    if len(points_px) < 2:
        return None
    unique = np.unique(np.round(points_px, 4), axis=0)
    if len(unique) < 2:
        return None
    if len(unique) == 2:
        p1, p2 = unique
        line = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    else:
        centroid = unique.mean(0)
        _, _, vh = np.linalg.svd(unique - centroid)
        direction = vh[0]
        normal = np.array([-direction[1], direction[0]])
        c = float(-normal @ centroid)
        line = np.array([normal[0], normal[1], c])
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-9:
        return None
    return line / norm


def line_intersection(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
    p = np.cross(l1, l2)
    if abs(float(p[2])) < 1e-9:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def angle_between_lines_deg(l1: np.ndarray, l2: np.ndarray) -> float:
    """Angle (0-90°) between two normalised homogeneous lines."""
    cos_a = abs(float(l1[0] * l2[0] + l1[1] * l2[1]))  # dot product of normals
    cos_a = min(cos_a, 1.0)
    angle_rad = math.acos(cos_a)
    angle_deg = math.degrees(angle_rad)
    # angle between lines is the complement if > 90
    return min(angle_deg, 180.0 - angle_deg)


def inside_image(pt: np.ndarray, w: int, h: int, margin: float = 100.0) -> bool:
    return -margin <= float(pt[0]) <= w + margin and -margin <= float(pt[1]) <= h + margin


def inside_pitch(pt: np.ndarray, eps: float = 1e-3) -> bool:
    return -eps <= float(pt[0]) <= PITCH_LENGTH + eps and -eps <= float(pt[1]) <= PITCH_WIDTH + eps


def project_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    out = cv2.perspectiveTransform(pts.astype(np.float32).reshape(-1, 1, 2), H)
    return out.reshape(-1, 2).astype(np.float64)


# ---------------------------------------------------------------------------
# Correspondence building
# ---------------------------------------------------------------------------

def _polyline_bbox_contains(pt: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray, margin: float = 150.0) -> bool:
    """
    Return True only if pt falls within the bounding box (+ margin) of BOTH polylines.
    This filters intersections that are geometrically valid but lie far from the
    annotated line segments — a common source of noisy correspondences.
    """
    x, y = float(pt[0]), float(pt[1])
    for pts in (pts_a, pts_b):
        min_x, min_y = pts.min(0) - margin
        max_x, max_y = pts.max(0) + margin
        if not (min_x <= x <= max_x and min_y <= y <= max_y):
            return False
    return True


def build_correspondences(
    annotation: dict,
    width: int,
    height: int,
    min_line_angle_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (image_points, field_points) arrays of shape (N, 2).

    Filters applied per line pair:
      1. Angle >= min_line_angle_deg (rejects near-parallel intersections).
      2. Intersection inside image bounds (+50px margin).
      3. Intersection inside the bounding box of both annotated polylines (+150px margin).
      4. Corresponding world intersection inside pitch bounds.
    """
    image_lines: Dict[str, np.ndarray] = {}
    raw_pixels: Dict[str, np.ndarray] = {}
    for raw_label, pts_raw in annotation.items():
        label = raw_label.strip()
        if label not in WORLD_LINES:
            continue
        pts_px = np.array(
            [(float(p["x"]) * width, float(p["y"]) * height) for p in pts_raw],
            dtype=np.float64,
        )
        line = fit_image_line(pts_px)
        if line is not None:
            image_lines[label] = line
            raw_pixels[label] = pts_px

    img_pts: List[np.ndarray] = []
    fld_pts: List[np.ndarray] = []

    labels = sorted(image_lines.keys())
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            # 1. Reject near-parallel pairs.
            angle = angle_between_lines_deg(image_lines[la], image_lines[lb])
            if angle < min_line_angle_deg:
                continue

            img_pt = line_intersection(image_lines[la], image_lines[lb])

            # 2. Allow a small margin: a valid corner can project just outside the frame.
            if img_pt is None or not inside_image(img_pt, width, height, margin=50.0):
                continue

            # 3. Must be near both polyline bounding boxes.
            if not _polyline_bbox_contains(img_pt, raw_pixels[la], raw_pixels[lb]):
                continue

            wla = np.asarray(WORLD_LINES[la], dtype=np.float64)
            wlb = np.asarray(WORLD_LINES[lb], dtype=np.float64)
            fld_pt = line_intersection(wla, wlb)

            # 4. World point inside pitch.
            if fld_pt is None or not inside_pitch(fld_pt):
                continue

            img_pts.append(img_pt)
            fld_pts.append(fld_pt)

    if img_pts:
        return np.array(img_pts), np.array(fld_pts)
    return np.empty((0, 2)), np.empty((0, 2))


# ---------------------------------------------------------------------------
# Homography estimation + non-linear refinement
# ---------------------------------------------------------------------------

def _h_residuals(h_flat: np.ndarray, img_pts: np.ndarray, fld_pts: np.ndarray) -> np.ndarray:
    """
    Symmetric reprojection residuals for Levenberg-Marquardt.

    Returns a vector of length 4*N:
        [dx_forward(0), dy_forward(0), dx_backward(0), dy_backward(0), ...]
    where forward maps img->field and backward maps field->img.
    """
    H = h_flat.reshape(3, 3)
    H = H / H[2, 2]  # keep scale

    fwd = project_points(img_pts, H) - fld_pts          # (N, 2) in metres
    H_inv = np.linalg.inv(H)
    bwd = project_points(fld_pts, H_inv) - img_pts      # (N, 2) in pixels

    # Scale backward residuals to metres so the loss is balanced.
    # Rough pixel-per-metre conversion: image is ~1920px wide over 105m -> ~18px/m
    PPM = 18.0
    return np.concatenate([fwd.ravel(), bwd.ravel() / PPM])


def estimate_homography(
    img_pts: np.ndarray,
    fld_pts: np.ndarray,
    ransac_thresh_m: float = 1.5,
    refine: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(img_pts) < 4:
        return None, None

    H, mask = cv2.findHomography(
        img_pts.astype(np.float32),
        fld_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_m,
        maxIters=10000,
        confidence=0.999,
    )
    if H is None or mask is None:
        return None, None

    if not refine:
        return H, mask

    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() < 4:
        return H, mask

    img_in = img_pts[inlier_mask]
    fld_in = fld_pts[inlier_mask]

    result = least_squares(
        _h_residuals,
        H.ravel(),
        args=(img_in, fld_in),
        method="lm",
        max_nfev=2000,
    )
    H_refined = result.x.reshape(3, 3)
    H_refined /= H_refined[2, 2]

    # Re-compute inlier mask with refined H (forward error only).
    fwd_err = np.linalg.norm(project_points(img_pts, H_refined) - fld_pts, axis=1)
    new_mask = (fwd_err < ransac_thresh_m * 2).astype(np.uint8).reshape(-1, 1)

    return H_refined, new_mask


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_errors(
    img_pts: np.ndarray,
    fld_pts: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (forward_errors_m, backward_errors_px) for inliers only."""
    inliers = mask.ravel().astype(bool)
    if not np.any(inliers):
        return np.empty(0), np.empty(0)
    fwd = np.linalg.norm(project_points(img_pts[inliers], H) - fld_pts[inliers], axis=1)
    H_inv = np.linalg.inv(H)
    bwd = np.linalg.norm(project_points(fld_pts[inliers], H_inv) - img_pts[inliers], axis=1)
    return fwd, bwd


def world_coverage(fld_pts: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """
    Return (x_range_m, y_range_m) of inlier world points.

    If the inlier correspondences all lie in a small portion of the pitch
    (e.g. only the right penalty area: x=[88.5, 105]), the homography is only
    constrained for that region and extrapolates wildly elsewhere.
    A minimum x_range of ~25m ensures at least two different x-positions are
    covered (e.g. penalty-box line + touchline intersection).
    """
    inliers = fld_pts[mask.ravel().astype(bool)]
    if len(inliers) < 2:
        return 0.0, 0.0
    return float(inliers[:, 0].max() - inliers[:, 0].min()), float(inliers[:, 1].max() - inliers[:, 1].min())


def inlier_spread_fraction(img_pts: np.ndarray, mask: np.ndarray, img_w: int, img_h: int) -> float:
    """
    Fraction of image area covered by the convex hull of inlier points.
    A homography fitted to spatially clustered points is ill-conditioned even if
    RANSAC reports low reprojection error — all points lying along a single line
    or bunched in one corner give a degenerate solution.
    Returns 0.0 if fewer than 3 inliers.
    """
    inliers = img_pts[mask.ravel().astype(bool)]
    if len(inliers) < 3:
        return 0.0
    hull = cv2.convexHull(inliers.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    return hull_area / (img_w * img_h)


def visible_field_bbox_span(H: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Project the 4 image corners via H (img→field) and return the (x_span_m, y_span_m)
    of the resulting bounding box in field metres.

    For a correct H covering a normal camera view:
      x_span is typically 20-120m, y_span is typically 30-100m.
    For a degenerate H (e.g. only right penalty area with wrong extrapolation):
      x_span can be hundreds of metres.
    """
    corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32)
    try:
        field_pts = cv2.perspectiveTransform(corners_img.reshape(-1, 1, 2), H).reshape(-1, 2)
    except cv2.error:
        return math.inf, math.inf
    if not np.all(np.isfinite(field_pts)):
        return math.inf, math.inf
    x_span = float(field_pts[:, 0].max() - field_pts[:, 0].min())
    y_span = float(field_pts[:, 1].max() - field_pts[:, 1].min())
    return x_span, y_span


def projected_pitch_span(H: np.ndarray, img_w: int, img_h: int) -> float:
    """
    Project the 4 pitch corners back into the image via H^{-1} and return the
    fraction of the image diagonal spanned by the convex hull of those 4 points.

    A degenerate homography maps all corners to a tiny cluster (low value),
    while a sensible wide-angle homography maps them across much of the image.
    Returns 0.0 if any projected corner is more than 3x the image size away.
    """
    corners_m = np.array([
        [0.0, 0.0], [PITCH_LENGTH, 0.0],
        [PITCH_LENGTH, PITCH_WIDTH], [0.0, PITCH_WIDTH],
    ], dtype=np.float64)
    H_inv = np.linalg.inv(H)
    projected = project_points(corners_m, H_inv)

    # Reject if any corner is wildly off-screen.
    max_dist = 3.0 * max(img_w, img_h)
    if np.any(np.abs(projected) > max_dist):
        return 0.0

    hull = cv2.convexHull(projected.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    img_diag_sq = float(img_w ** 2 + img_h ** 2)
    return hull_area / img_diag_sq  # normalise by diag² for scale-independence


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _polyline_field(overlay: np.ndarray, H_inv: np.ndarray, pts_m: List[Tuple[float, float]], color, thickness=2) -> None:
    """
    Project field-coordinate polyline back into image and draw it.
    Splits the polyline at segments where both endpoints project far outside the image,
    preventing diagonal lines flying across the frame when a corner is off-screen.
    """
    h, w = overlay.shape[:2]
    pts = np.array(pts_m, dtype=np.float64)
    img_pts = project_points(pts, H_inv)

    # Draw segment-by-segment; skip segments where both endpoints are wildly outside.
    far = 30.0  # pixels beyond image edge considered "too far"
    def _in_range(p: np.ndarray) -> bool:
        return -far <= float(p[0]) <= w + far and -far <= float(p[1]) <= h + far

    run: List[np.ndarray] = []
    for p in img_pts:
        if _in_range(p):
            run.append(p)
        else:
            if len(run) >= 2:
                seg = np.round(np.array(run)).astype(np.int32)
                cv2.polylines(overlay, [seg.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            run = []
    if len(run) >= 2:
        seg = np.round(np.array(run)).astype(np.int32)
        cv2.polylines(overlay, [seg.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _circle_pts(cx: float, cy: float, r: float, n: int = 120) -> List[Tuple[float, float]]:
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n)]


def _visible_field_polygon(H: np.ndarray, img_w: int, img_h: int) -> Optional[np.ndarray]:
    """
    Project the 4 image corners via H (img→field) to get the pitch region
    that is visible in this frame.  Returns a (4,2) float32 polygon in field
    metres, or None if the projection fails.
    """
    corners_img = np.array(
        [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=np.float32
    )
    field_pts = cv2.perspectiveTransform(corners_img.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.all(np.isfinite(field_pts)):
        return None
    return field_pts


def _in_visible_region(pt_m: Tuple[float, float], vis_poly: Optional[np.ndarray], margin_m: float = 5.0) -> bool:
    """Return True if field point pt_m is inside (or within margin_m of) vis_poly."""
    if vis_poly is None:
        return True
    x, y = float(pt_m[0]), float(pt_m[1])
    # Grow the polygon by margin in x/y for line segments that approach the border.
    hull = cv2.convexHull(vis_poly)
    dist = cv2.pointPolygonTest(hull, (x, y), measureDist=True)
    return dist >= -margin_m


def draw_pitch_overlay(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    H_inv = np.linalg.inv(H)
    h, w = image.shape[:2]
    W = (255, 255, 255)
    Y = (0, 220, 255)

    vis_poly = _visible_field_polygon(H, w, h)

    PL, PW = PITCH_LENGTH, PITCH_WIDTH

    def _draw(pts, color, thickness=2):
        # Filter to only include consecutive runs of points visible in the camera.
        visible = [_in_visible_region(p, vis_poly) for p in pts]
        run = []
        for p, v in zip(pts, visible):
            if v:
                run.append(p)
            else:
                if len(run) >= 2:
                    _polyline_field(overlay, H_inv, run, color, thickness)
                run = []
        if len(run) >= 2:
            _polyline_field(overlay, H_inv, run, color, thickness)

    _draw([(0,0),(PL,0),(PL,PW),(0,PW),(0,0)], W)
    _draw([(CENTER_X, 0), (CENTER_X, PW)], W)

    for xs, xe in [(0, PENALTY_DEPTH), (PL, PL - PENALTY_DEPTH)]:
        _draw([(xs, PENALTY_Y_TOP),(xe, PENALTY_Y_TOP),(xe, PENALTY_Y_BOTTOM),(xs, PENALTY_Y_BOTTOM),(xs, PENALTY_Y_TOP)], W)
    for xs, xe in [(0, GOAL_AREA_DEPTH), (PL, PL - GOAL_AREA_DEPTH)]:
        _draw([(xs, GOAL_AREA_Y_TOP),(xe, GOAL_AREA_Y_TOP),(xe, GOAL_AREA_Y_BOTTOM),(xs, GOAL_AREA_Y_BOTTOM),(xs, GOAL_AREA_Y_TOP)], W)

    circle_pts_m = _circle_pts(CENTER_X, CENTER_Y, CENTER_CIRCLE_RADIUS)
    if any(_in_visible_region(p, vis_poly, margin_m=2.0) for p in circle_pts_m):
        _draw(circle_pts_m, Y)

    for fx, fy in [(11.0, CENTER_Y), (CENTER_X, CENTER_Y), (PL - 11.0, CENTER_Y)]:
        if _in_visible_region((fx, fy), vis_poly, margin_m=1.0):
            p = project_points(np.array([[fx, fy]]), H_inv)[0].astype(int)
            if 0 <= p[0] < w and 0 <= p[1] < h:
                cv2.circle(overlay, tuple(p), 5, Y, -1, lineType=cv2.LINE_AA)

    return overlay


def draw_correspondence_overlay(
    image: np.ndarray,
    img_pts: np.ndarray,
    fld_pts: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Green filled circle  = inlier image point.
    Blue open circle     = expected position (where H maps the world point back).
    Cyan line            = residual between annotated and expected (inliers only).
    Red circle           = outlier (RANSAC-rejected, not used for H).
    """
    overlay = image.copy()
    H_inv = np.linalg.inv(H)
    inliers = mask.ravel().astype(bool)
    h, w = image.shape[:2]

    for ip, fp, is_inlier in zip(img_pts, fld_pts, inliers):
        ip_i = tuple(np.round(ip).astype(int))
        if is_inlier:
            expected_img = project_points(fp.reshape(1, 2), H_inv)[0]
            exp_i = tuple(np.round(expected_img).astype(int))
            cv2.line(overlay, ip_i, exp_i, (0, 255, 255), 1)
            cv2.circle(overlay, exp_i, 6, (255, 130, 0), 2)   # blue ring = expected
            cv2.circle(overlay, ip_i, 5, (0, 210, 0), -1)     # green dot = annotated
        else:
            cv2.circle(overlay, ip_i, 6, (0, 0, 220), 2)      # red ring = outlier

    # Legend (bottom-left)
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    legends = [
        ((0, 210, 0),   "inlier (annotated)"),
        ((255, 130, 0), "inlier (expected via H)"),
        ((0, 0, 220),   "outlier (RANSAC rejected)"),
    ]
    for i, (color, text) in enumerate(legends):
        y = h - 20 - i * 22
        cv2.circle(overlay, (18, y), 6, color, -1 if color == (0, 210, 0) else 2)
        cv2.putText(overlay, text, (30, y + 5), font, scale, (220, 220, 220), th, cv2.LINE_AA)

    return overlay


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    frame_id: str
    status: str
    num_correspondences: int = 0
    num_inliers: int = 0
    fwd_error_mean_m: float = math.nan
    fwd_error_median_m: float = math.nan
    bwd_error_mean_px: float = math.nan
    bwd_error_median_px: float = math.nan
    inlier_spread: float = math.nan
    bwd_error_max_px: float = math.nan
    H: Optional[np.ndarray] = field(default=None, repr=False)


def process_frame(
    dataset: Path,
    frame_id: str,
    output_dir: Path,
    debug_overlays: bool,
    min_points: int,
    ransac_thresh_m: float,
    min_line_angle_deg: float,
    max_fwd_error_m: float,
    min_spread_fraction: float = 0.01,
) -> FrameResult:
    ann_path = dataset / "annotations" / f"{frame_id}.json"
    img_path = find_image_path(dataset / "images", frame_id)

    if img_path is None or not ann_path.exists():
        return FrameResult(frame_id, "missing_image_or_annotation")

    image = cv2.imread(str(img_path))
    if image is None:
        return FrameResult(frame_id, "cannot_read_image")

    h, w = image.shape[:2]
    annotation = read_annotation(ann_path)
    img_pts, fld_pts = build_correspondences(annotation, w, h, min_line_angle_deg)

    if len(img_pts) < min_points:
        return FrameResult(frame_id, "not_enough_correspondences", len(img_pts))

    H, mask = estimate_homography(img_pts, fld_pts, ransac_thresh_m, refine=True)
    if H is None:
        return FrameResult(frame_id, "homography_failed", len(img_pts))

    fwd_errs, bwd_errs = compute_errors(img_pts, fld_pts, H, mask)
    num_inliers = int(mask.sum())
    spread = inlier_spread_fraction(img_pts, mask, w, h)
    vis_x_span, vis_y_span = visible_field_bbox_span(H, w, h)

    if num_inliers < min_points:
        status = "not_enough_inliers"
    elif spread < min_spread_fraction:
        status = "low_spatial_spread"
    elif vis_x_span > 150.0 or vis_y_span > 120.0:
        # Image corners project >150m/120m in field coords → wildly degenerate H.
        # A correct H spans at most ~105×68m; huge values mean the correspondences
        # only covered one end of the pitch and H extrapolates badly elsewhere.
        status = "degenerate_homography"
    elif len(fwd_errs) and float(np.mean(fwd_errs)) > max_fwd_error_m:
        status = "high_reprojection_error"
    elif len(bwd_errs) and float(np.max(bwd_errs)) > 10.0:
        status = "degenerate_homography"
    else:
        status = "ok"

    if debug_overlays and status == "ok":
        overlay_dir = output_dir / "debug_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        pitch_ov = draw_pitch_overlay(image, H)
        corr_ov = draw_correspondence_overlay(image, img_pts, fld_pts, H, mask)
        cv2.imwrite(str(overlay_dir / f"{frame_id}_pitch.jpg"), pitch_ov)
        cv2.imwrite(str(overlay_dir / f"{frame_id}_corr.jpg"), corr_ov)

    return FrameResult(
        frame_id=frame_id,
        status=status,
        num_correspondences=len(img_pts),
        num_inliers=num_inliers,
        fwd_error_mean_m=float(np.mean(fwd_errs)) if len(fwd_errs) else math.nan,
        fwd_error_median_m=float(np.median(fwd_errs)) if len(fwd_errs) else math.nan,
        bwd_error_mean_px=float(np.mean(bwd_errs)) if len(bwd_errs) else math.nan,
        bwd_error_median_px=float(np.median(bwd_errs)) if len(bwd_errs) else math.nan,
        inlier_spread=spread,
        bwd_error_max_px=float(np.max(bwd_errs)) if len(bwd_errs) else math.nan,

        H=H if status == "ok" else None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer pitch homography — v4")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=str, default=None, help="Process a single frame id")
    parser.add_argument("--debug-overlays", action="store_true")
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--ransac-thresh-m", type=float, default=1.5,
                        help="RANSAC inlier threshold in field metres (default 1.5)")
    parser.add_argument("--min-line-angle-deg", type=float, default=15.0,
                        help="Minimum angle between two pitch lines to use their intersection (default 15°)")
    parser.add_argument("--max-fwd-error-m", type=float, default=3.0,
                        help="Maximum mean forward reprojection error in metres to accept a frame (default 3.0)")
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "homographies").mkdir(parents=True, exist_ok=True)

    if args.frame:
        frame_ids = [args.frame]
    else:
        frame_ids = sorted(
            p.stem for p in (args.dataset / "annotations").glob("*.json")
            if not p.name.startswith("._")
        )

    results: List[FrameResult] = []
    for fid in frame_ids:
        r = process_frame(
            args.dataset, fid, output_dir,
            args.debug_overlays,
            args.min_points,
            args.ransac_thresh_m,
            args.min_line_angle_deg,
            args.max_fwd_error_m,
        )
        results.append(r)
        if r.H is not None:
            np.savetxt(output_dir / "homographies" / f"{fid}_image_to_field.txt", r.H, fmt="%.10f")
        print(
            f"{fid}  [{r.status}]  "
            f"corr={r.num_correspondences}  inliers={r.num_inliers}  "
            f"fwd={r.fwd_error_mean_m:.3f}m  bwd={r.bwd_error_mean_px:.1f}px"
        )

    # Save all good homographies as a single .npz for convenience.
    good = {r.frame_id: r.H for r in results if r.H is not None}
    if good:
        np.savez(output_dir / "homographies.npz", **good)

    csv_path = output_dir / "homography_summary.csv"
    fieldnames = [
        "frame_id", "status",
        "num_correspondences", "num_inliers",
        "fwd_error_mean_m", "fwd_error_median_m",
        "bwd_error_mean_px", "bwd_error_median_px",
        "inlier_spread", "bwd_error_max_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})

    ok = sum(r.status == "ok" for r in results)
    print(f"\nProcessed: {len(results)}  OK: {ok}  Failed: {len(results) - ok}")
    print(f"Homographies (.npz): {output_dir / 'homographies.npz'}")
    print(f"Summary CSV:         {csv_path}")
    if args.debug_overlays:
        print(f"Debug overlays:      {output_dir / 'debug_overlays'}")


if __name__ == "__main__":
    main()
