"""
Homography Pipeline: Video + Tracking Parquet -> 2D Top-View Koordinaten
========================================================================
Masterarbeit: Computer-Vision-based Extraction of Player and Ball Trajectories

Voraussetzungen:
    pip install opencv-python pandas numpy scipy tqdm matplotlib

PnLCalib muss installiert sein:
    git clone https://github.com/mguti97/PnLCalib
    cd PnLCalib
    # Gewichte herunterladen (siehe PnLCalib README)

Verwendung:
    1. Schnelltest (30 Sekunden Clip):
       python homography_pipeline.py --video clip.mp4 --parquet tracking.parquet --sample_every 5

    2. Voller Lauf (66 Minuten):
       python homography_pipeline.py --video full_match.mp4 --parquet tracking.parquet --sample_every 10

    3. Nur Visualisierung (wenn Homographien schon berechnet):
       python homography_pipeline.py --parquet tracking_2d.parquet --visualize_only
"""

import os
import yaml
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as tvf
from PIL import Image as PILImage
import sys
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# config.py liegt unter /home/manser/projects/dat24-masterarbeit-manser/
CONFIG_PATH = Path(__file__).resolve().parents[2]
if str(CONFIG_PATH) not in sys.path:
    sys.path.insert(0, str(CONFIG_PATH))

from config import (
    PNLCALIB_REPO,
    PNLCALIB_WEIGHTS_KP,
    PNLCALIB_WEIGHTS_LINES,
    TRACKING_DIR,
    VIDEOS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Standard FIFA Platzmasse in Metern (105 x 68)
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# 2D Template Aufloesung in Pixeln fuer die Top-View Ausgabe
TEMPLATE_WIDTH_PX  = 1050
TEMPLATE_HEIGHT_PX = 680

# Meter zu Pixel Skalierung
SCALE_X = TEMPLATE_WIDTH_PX  / PITCH_LENGTH_M
SCALE_Y = TEMPLATE_HEIGHT_PX / PITCH_WIDTH_M

# Confidence-Schwelle: Frames unter diesem Wert werden verworfen
CONFIDENCE_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# 2. FRAME EXTRAKTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, output_dir: str, sample_every: int = 5) -> dict:
    """
    Extrahiert jeden N-ten Frame aus dem Video.

    Returns:
        frame_map: {frame_idx: file_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_s   = total_frames / fps

    print(f"[Frame Extraktion]")
    print(f"  Video:         {video_path}")
    print(f"  Frames gesamt: {total_frames} ({duration_s/60:.1f} Min @ {fps:.1f} fps)")
    print(f"  Sample-Rate:   jeder {sample_every}. Frame -> ~{total_frames//sample_every} Frames")

    frame_map = {}
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Extrahiere Frames", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            path = output_dir / f"frame_{frame_idx:07d}.jpg"
            cv2.imwrite(str(path), frame)
            frame_map[frame_idx] = str(path)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"  Gespeichert:   {len(frame_map)} Frames in '{output_dir}'\n")
    return frame_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. PNLCALIB INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def cam_params_to_homography(final_params_dict: dict,
                              img_w: int = 1280,
                              img_h: int = 720) -> np.ndarray:
    """
    Berechnet eine 3x3 Homography (Bild -> Template-Pixel) aus den
    PnLCalib Kameraparametern.

    Strategie: Projiziere bekannte Feldpunkte (Meter) ins Bild via P,
    sammle Korrespondenzpaare (Bildpixel <-> Template-Pixel) und
    berechne H mit findHomography. Das ist robuster als die analytische
    Inversion von H_proj.

    Feldpunkte in PnLCalib-Weltkoordinaten (Ursprung = Platzmitte):
        x_pnl = x_m - 52.5,  y_pnl = y_m - 34,  z = 0
    """
    try:
        cam = final_params_dict["cam_params"]
        fx  = cam["x_focal_length"]
        fy  = cam["y_focal_length"]
        pp  = np.array(cam["principal_point"])
        pos = np.array(cam["position_meters"])
        R   = np.array(cam["rotation_matrix"])

        It = np.eye(4)[:-1]
        It[:, -1] = -pos
        Q = np.array([[fx, 0, pp[0]],
                      [0, fy, pp[1]],
                      [0,  0,     1]])
        P = Q @ (R @ It)

        # Bekannte Feldpunkte (x_m, y_m) in Metern
        PITCH_KEYPOINTS_M = np.array([
            [0.0,    0.0],    # linke untere Ecke
            [105.0,  0.0],    # rechte untere Ecke
            [0.0,   68.0],    # linke obere Ecke
            [105.0, 68.0],    # rechte obere Ecke
            [52.5,   0.0],    # Mittellinie unten
            [52.5,  68.0],    # Mittellinie oben
            [52.5,  34.0],    # Mittelpunkt
            [16.5,  13.84],   # linker Strafraum unten
            [16.5,  54.16],   # linker Strafraum oben
            [88.5,  13.84],   # rechter Strafraum unten
            [88.5,  54.16],   # rechter Strafraum oben
            [0.0,   24.84],   # linker Torraum unten
            [0.0,   43.16],   # linker Torraum oben
            [11.0,  34.0],    # linker Elfmeterpunkt
            [94.0,  34.0],    # rechter Elfmeterpunkt
        ])

        src_pts = []   # Bildpixel
        dst_pts = []   # Template-Pixel

        for (x_m, y_m) in PITCH_KEYPOINTS_M:
            # PnLCalib-zentrierte Weltkoordinaten
            world = np.array([x_m - 52.5, y_m - 34.0, 0.0, 1.0])
            img_hom = P @ world
            if abs(img_hom[2]) < 1e-6:
                continue
            img_hom /= img_hom[2]
            u, v = float(img_hom[0]), float(img_hom[1])

            # Nur Punkte behalten die (mit Puffer) im Bild liegen
            margin = 0.3
            if (-img_w * margin < u < img_w * (1 + margin) and
                    -img_h * margin < v < img_h * (1 + margin)):
                src_pts.append([u, v])
                dst_pts.append([x_m * SCALE_X, y_m * SCALE_Y])

        if len(src_pts) < 4:
            return None

        src_np = np.array(src_pts, dtype=np.float32)
        dst_np = np.array(dst_pts, dtype=np.float32)
        H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)

        if H is None or mask is None or mask.sum() < 4:
            return None

        return H.astype(np.float64)

    except Exception as e:
        print(f"  Homography-Berechnung fehlgeschlagen: {e}")
        return None


def run_pnlcalib(frames_dir: str, output_pkl: str,
                 weights_kp: str = PNLCALIB_WEIGHTS_KP,
                 weights_line: str = PNLCALIB_WEIGHTS_LINES,
                 device: str = "cuda:0",
                 kp_threshold: float = 0.3434,
                 line_threshold: float = 0.7867,
                 pnl_refine: bool = True) -> dict:
    """
    Laedt PnLCalib-Modelle einmalig und verarbeitet alle Frames im frames_dir.
    Gibt {frame_idx: H_matrix (3x3)} zurueck.
    Frames bei denen PnLCalib versagt (z.B. kein Feld sichtbar) werden uebersprungen.
    """

    sys.path.insert(0, PNLCALIB_REPO)

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
        complete_keypoints,
        coords_to_dict,
    )
    import torchvision.transforms.functional as tvf
    from PIL import Image as PILImage

    cfg_path   = Path(PNLCALIB_REPO) / "config" / "hrnetv2_w48.yaml"
    cfg_l_path = Path(PNLCALIB_REPO) / "config" / "hrnetv2_w48_l.yaml"

    cfg   = yaml.safe_load(open(cfg_path, "r"))
    cfg_l = yaml.safe_load(open(cfg_l_path, "r"))

    print(f"[PnLCalib] Lade Modelle auf {device}...")
    loaded_state = torch.load(weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform_resize = T.Resize((540, 960))

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    print(f"[PnLCalib] Verarbeite {len(frame_files)} Frames...")

    homographies = {}
    failed = 0

    for frame_path in tqdm(frame_files, desc="PnLCalib Inference"):
        frame_idx = int(frame_path.stem.split("_")[-1])

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = frame_bgr.shape[:2]

        pil_frame = PILImage.fromarray(frame_rgb)
        tensor = tvf.to_tensor(pil_frame).float().unsqueeze(0)
        if tensor.size()[-1] != 960:
            tensor = transform_resize(tensor)
        tensor = tensor.to(device)

        _, _, h, w = tensor.size()

        cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)

        with torch.no_grad():
            heatmaps   = model(tensor)
            heatmaps_l = model_l(tensor)

        kp_coords   = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:, :-1, :, :])
        kp_dict     = coords_to_dict(kp_coords, threshold=kp_threshold)
        lines_dict  = coords_to_dict(line_coords, threshold=line_threshold)
        kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

        cam.update(kp_dict, lines_dict)
        final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine)

        if final_params_dict is None:
            failed += 1
            continue

        H = cam_params_to_homography(final_params_dict, img_w=w_orig, img_h=h_orig)
        if H is not None:
            homographies[frame_idx] = H

    print(f"[PnLCalib] Fertig: {len(homographies)} Homographien, {failed} Frames uebersprungen")

    with open(output_pkl, "wb") as pkl_f:
        pickle.dump(homographies, pkl_f)
    print(f"[PnLCalib] Gespeichert: {output_pkl}\n")
    return homographies


def _manual_annotation_fallback(frames_dir: str) -> dict:
    """
    Fallback: Manuelles Annotieren von Keypoints für ausgewählte Frames.
    Öffnet ein Matplotlib-Fenster zum Klicken der Punktkorrespondenzen.

    Keypoints auf dem Platz (in Metern, Ursprung = linke untere Ecke):
        Punkt  1: linke untere Eckfahne        (  0,  0)
        Punkt  2: linke obere Eckfahne         (  0, 68)
        Punkt  3: rechte untere Eckfahne       (105,  0)
        Punkt  4: rechte obere Eckfahne        (105, 68)
        Punkt  5: Mittelpunkt Mittellinie unten (52.5,  0)
        Punkt  6: Mittelpunkt Mittellinie oben  (52.5, 68)
        Punkt  7: Mittelpunkt                  (52.5, 34)
        Punkt  8: linker Strafraumecke unten   ( 16.5, 13.84)
        Punkt  9: linker Strafraumecke oben    ( 16.5, 54.16)
        Punkt 10: rechter Strafraumecke unten  ( 88.5, 13.84)
        Punkt 11: rechter Strafraumecke oben   ( 88.5, 54.16)
    """

    # Weltkoordinaten der Keypoints (Meter, Ursprung links unten)
    KEYPOINTS_WORLD = np.array([
        [  0.0,  0.0],   # 1
        [  0.0, 68.0],   # 2
        [105.0,  0.0],   # 3
        [105.0, 68.0],   # 4
        [ 52.5,  0.0],   # 5
        [ 52.5, 68.0],   # 6
        [ 52.5, 34.0],   # 7
        [ 16.5, 13.84],  # 8
        [ 16.5, 54.16],  # 9
        [ 88.5, 13.84],  # 10
        [ 88.5, 54.16],  # 11
    ], dtype=np.float32)

    # Weltkoordinaten -> Template-Pixel
    dst_points = np.column_stack([
        KEYPOINTS_WORLD[:, 0] * SCALE_X,
        KEYPOINTS_WORLD[:, 1] * SCALE_Y,
    ]).astype(np.float32)

    KEYPOINT_LABELS = [
        "1: linke untere Eckfahne (0,0)",
        "2: linke obere Eckfahne (0,68)",
        "3: rechte untere Eckfahne (105,0)",
        "4: rechte obere Eckfahne (105,68)",
        "5: Mittellinie unten (52.5,0)",
        "6: Mittellinie oben (52.5,68)",
        "7: Mittelpunkt (52.5,34)",
        "8: linker Strafraum unten (16.5,13.84)",
        "9: linker Strafraum oben (16.5,54.16)",
        "10: rechter Strafraum unten (88.5,13.84)",
        "11: rechter Strafraum oben (88.5,54.16)",
    ]

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    # Nimm gleichmäßig verteilte Frames zur Annotation
    n_annotate = min(5, len(frame_files))
    selected = [frame_files[i * len(frame_files) // n_annotate] for i in range(n_annotate)]

    homographies = {}

    for frame_path in selected:
        frame_idx = int(frame_path.stem.split("_")[-1])
        img = cv2.imread(str(frame_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(f"\n[Annotation] Frame {frame_idx}: {frame_path.name}")
        print("  Klicke die folgenden Punkte IN DIESER REIHENFOLGE (mind. 4):")
        for label in KEYPOINT_LABELS:
            print(f"    {label}")
        print("  Drücke ENTER wenn fertig, BACKSPACE zum Rückgängig machen.")

        clicked = []

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(img_rgb)
        ax.set_title(f"Frame {frame_idx} – Klicke Keypoints (min. 4)\nReihenfolge: {', '.join([l.split(':')[0] for l in KEYPOINT_LABELS])}", fontsize=9)

        scatter = ax.scatter([], [], c='red', s=80, zorder=5)
        texts   = []

        def onclick(event):
            if event.inaxes != ax:
                return
            idx = len(clicked)
            if idx >= len(KEYPOINT_LABELS):
                return
            clicked.append((event.xdata, event.ydata))
            xs = [p[0] for p in clicked]
            ys = [p[1] for p in clicked]
            scatter.set_offsets(np.c_[xs, ys])
            t = ax.text(event.xdata + 5, event.ydata - 5,
                        str(idx + 1), color='yellow', fontsize=8, fontweight='bold')
            texts.append(t)
            fig.canvas.draw()

        def onkey(event):
            if event.key == 'enter':
                plt.close()
            elif event.key == 'backspace' and clicked:
                clicked.pop()
                if texts:
                    texts.pop().remove()
                xs = [p[0] for p in clicked]
                ys = [p[1] for p in clicked]
                scatter.set_offsets(np.c_[xs, ys] if xs else np.empty((0, 2)))
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        plt.tight_layout()
        plt.show()

        if len(clicked) < 4:
            print(f"  Zu wenige Punkte ({len(clicked)}) – Frame wird übersprungen.")
            continue

        n = len(clicked)
        src = np.array(clicked[:n], dtype=np.float32)
        dst = dst_points[:n]

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is not None:
            inliers = int(mask.sum())
            print(f"  Homography berechnet – {inliers}/{n} Inlier")
            homographies[frame_idx] = H
        else:
            print(f"  findHomography fehlgeschlagen – Frame übersprungen.")

    return homographies


# ─────────────────────────────────────────────────────────────────────────────
# 4. HOMOGRAPHY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_homography_valid(homographies: dict,
                            img_w: int = 1280,
                            img_h: int = 720) -> dict:
    """
    Behaelt nur Homographien bei denen typische Bildpunkte
    auf plausible Platzmeter-Koordinaten projizieren.

    Prueft 9 gleichmaessig verteilte Bildpunkte (3x3 Grid).
    Verwirft Homographien deren Projektion stark ausserhalb
    des Feldes liegt (grosser Puffer von 30m erlaubt Randpositionen).
    """
    # 3x3 Grid ueber das Bild verteilt
    test_points = np.array([
        [[img_w * 0.25, img_h * 0.5]],
        [[img_w * 0.5,  img_h * 0.5]],
        [[img_w * 0.75, img_h * 0.5]],
        [[img_w * 0.25, img_h * 0.75]],
        [[img_w * 0.5,  img_h * 0.75]],
        [[img_w * 0.75, img_h * 0.75]],
        [[img_w * 0.25, img_h * 0.9]],
        [[img_w * 0.5,  img_h * 0.9]],
        [[img_w * 0.75, img_h * 0.9]],
    ], dtype=np.float32)

    filtered = {}
    for frame_idx, H in homographies.items():
        try:
            mapped = cv2.perspectiveTransform(test_points, H)
            # Umrechnung Template-Pixel -> Meter
            xs_m = mapped[:, 0, 0] / SCALE_X
            ys_m = mapped[:, 0, 1] / SCALE_Y
            # Grosszuegiger Puffer: Platz ist 0-105m x 0-68m
            if (xs_m.min() > -30 and xs_m.max() < 135 and
                    ys_m.min() > -20 and ys_m.max() < 88):
                filtered[frame_idx] = H
        except Exception:
            continue

    removed = len(homographies) - len(filtered)
    print(f"[Filter] {removed} ungueltige Homographien entfernt, {len(filtered)} behalten\n")
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# 5. HOMOGRAPHY INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────
# 4. HOMOGRAPHY INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def detect_camera_segments(homographies: dict,
                            max_center_shift_px: float = 400.0,
                            min_segment_keyframes: int = 2) -> list:
    """
    Erkennt Kamerawechsel anhand der Template-Position des Bildmittelpunkts.

    H bildet Bildpixel -> Template-Pixel ab. Wir projizieren daher einen festen
    Bildpunkt (Bildmitte) durch H und messen dessen Verschiebung im Template.
    Ein grosser Sprung bedeutet Kamerawechsel.

    Segmente mit weniger als min_segment_keyframes Stützpunkten werden verworfen
    und mit dem vorherigen Segment zusammengeführt — einzelne Ausreisser-Frames
    sollen keinen eigenen Schnitt erzeugen.

    max_center_shift_px:    Schwellwert in Template-Pixeln (Standard: 400)
    min_segment_keyframes:  Mindestanzahl Keyframes pro Segment (Standard: 2)
    """
    indices = sorted(homographies.keys())

    img_w_ref = 1920.0
    img_h_ref = 1080.0
    img_center = np.array([[[img_w_ref / 2.0, img_h_ref / 2.0]]], dtype=np.float32)

    # Rohe Segmentgrenzen ermitteln
    raw_segments = []
    seg_start = indices[0]
    prev_pos = cv2.perspectiveTransform(img_center, homographies[indices[0]])[0][0]

    for i in range(1, len(indices)):
        idx = indices[i]
        curr_pos = cv2.perspectiveTransform(img_center, homographies[idx])[0][0]
        shift_px = float(np.linalg.norm(curr_pos - prev_pos))

        if shift_px > max_center_shift_px:
            raw_segments.append((seg_start, indices[i - 1]))
            seg_start = idx

        prev_pos = curr_pos

    raw_segments.append((seg_start, indices[-1]))

    # Segmente mit zu wenigen Keyframes mit dem Vorgänger zusammenführen
    segments = []
    for seg in raw_segments:
        seg_keys = [k for k in indices if seg[0] <= k <= seg[1]]
        if len(seg_keys) < min_segment_keyframes and segments:
            # Mit Vorgänger zusammenführen
            prev = segments.pop()
            segments.append((prev[0], seg[1]))
        else:
            segments.append(seg)

    print(f"[Segmentierung] {len(segments)} Kamera-Segmente erkannt")
    for seg in segments:
        n_keys = len([k for k in indices if seg[0] <= k <= seg[1]])
        print(f"  Frame {seg[0]:5d} – {seg[1]:5d}  ({n_keys} Keyframes)")
    return segments


def build_interpolator(homographies: dict, total_frames: int):
    """
    Erstellt eine Interpolationsfunktion H(frame_idx) -> 3x3 Matrix.

    Strategie:
    - Kameraschnittpunkte werden erkannt und nicht ueberbrückt
    - Innerhalb jedes Segments: lineare Interpolation
    - Ausserhalb aller Segmente: naechste gueltige H (carry-forward/backward)
    """
    indices = sorted(homographies.keys())

    if len(indices) == 0:
        raise ValueError("Keine gueltigen Homographien vorhanden!")

    if len(indices) == 1:
        H_single = homographies[indices[0]]
        print("[Interpolation] Nur 1 Homographie – wird fuer alle Frames verwendet.")
        return lambda _: H_single

    # Kamera-Segmente erkennen
    segments = detect_camera_segments(homographies)

    # Pro Segment einen Interpolator erstellen
    seg_interpolators = []
    for seg_start, seg_end in segments:
        seg_indices = [i for i in indices if seg_start <= i <= seg_end]
        if len(seg_indices) < 2:
            H_seg = homographies[seg_indices[0]]
            seg_interpolators.append((seg_start, seg_end, None, H_seg))
            continue
        H_flat = np.array([homographies[i].flatten() for i in seg_indices])
        fn = interp1d(seg_indices, H_flat, axis=0, kind="linear",
                      bounds_error=False,
                      fill_value=(H_flat[0], H_flat[-1]))
        seg_interpolators.append((seg_start, seg_end, fn, None))

    coverage = (indices[-1] - indices[0]) / max(total_frames, 1) * 100
    print(f"[Interpolation] {len(indices)} Stuetzpunkte, "
          f"Frame {indices[0]}-{indices[-1]} ({coverage:.1f}% Abdeckung)")

    # Fallback: alle H-Werte fuer Nearest-Neighbor ausserhalb
    H_all_flat = np.array([homographies[i].flatten() for i in indices])
    nn_fn = interp1d(indices, H_all_flat, axis=0, kind="nearest",
                     bounds_error=False,
                     fill_value=(H_all_flat[0], H_all_flat[-1]))

    def get_H(frame_idx: int) -> np.ndarray:
        f = float(frame_idx)
        # Suche passendes Segment
        for seg_start, seg_end, fn, H_const in seg_interpolators:
            if seg_start <= f <= seg_end:
                if H_const is not None:
                    return H_const
                H = fn(f).reshape(3, 3)
                if H[2, 2] != 0:
                    H = H / H[2, 2]
                return H.astype(np.float64)
        # Ausserhalb aller Segmente: Nearest-Neighbor
        H = nn_fn(f).reshape(3, 3)
        if H[2, 2] != 0:
            H = H / H[2, 2]
        return H.astype(np.float64)

    return get_H


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRACKING KOORDINATEN TRANSFORMIEREN
# ─────────────────────────────────────────────────────────────────────────────

def transform_tracking(df: pd.DataFrame, get_H, total_frames: int) -> pd.DataFrame:
    """
    Wendet die frame-spezifische Homography auf alle Tracking-Koordinaten an.

    Erwartet im DataFrame:
        - frame_idx (int): Frame-Nummer
        - x (float):       Pixel X im Originalbild
        - y (float):       Pixel Y im Originalbild

    Optional:
        - track_id, class_id, confidence (werden durchgereicht)

    Fügt hinzu:
        - x_2d (float): X in Template-Pixeln (0 = linke Torlinie)
        - y_2d (float): Y in Template-Pixeln (0 = untere Seitenlinie)
        - x_m  (float): X in Metern
        - y_m  (float): Y in Metern
    """
    print(f"[Transform] {len(df)} Tracking-Einträge werden transformiert...")

    # ── Spalten-Mapping für tracking_clean.parquet ──────────────────────────
    df = df.rename(columns={"frame": "frame_idx"})
    df["x"] = df["x_center"]

    # Für Spieler: y_max (Füsse = unterster Punkt der Bounding Box) verwenden,
    # da die Homography den Bodenpunkt abbildet.
    # Für den Ball: y_center (Ball ist klein, Mittelpunkt ≈ Bodenkontakt).
    if "object_type" in df.columns and "y_max" in df.columns:
        df["y"] = np.where(df["object_type"] == "player", df["y_max"], df["y_center"])
        print(f"[Transform] Spieler: y_max (Füsse), Ball: y_center")
    else:
        df["y"] = df["y_center"]

    required = {"frame_idx", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten im Parquet: {missing}\n"
                         f"Vorhandene Spalten: {list(df.columns)}")

    # Vorberechnung: eine H pro eindeutigem Frame
    unique_frames = df["frame_idx"].unique()
    H_cache = {int(fi): get_H(int(fi)) for fi in tqdm(unique_frames, desc="H-Matrizen cachen")}

    # Batch-Transformation: alle Punkte eines Frames gemeinsam projizieren.
    sorted_keys = sorted(H_cache.keys())
    x2d_out = np.empty(len(df), dtype=np.float64)
    y2d_out = np.empty(len(df), dtype=np.float64)

    for fi, group in tqdm(df.groupby("frame_idx"), desc="Punkte transformieren"):
        fi = int(fi)
        if fi not in H_cache:
            # Nearest-neighbor Fallback
            nearest = sorted_keys[int(np.argmin(np.abs(np.array(sorted_keys) - fi)))]
            H = H_cache[nearest]
        else:
            H = H_cache[fi]

        pts = group[["x", "y"]].values.astype(np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x2d_out[group.index] = mapped[:, 0]
        y2d_out[group.index] = mapped[:, 1]

    df = df.copy()
    df["x_2d"] = x2d_out
    df["y_2d"] = y2d_out
    df["x_m"]  = x2d_out / SCALE_X
    # PnLCalib: y=0 ist die Ferne Seitenlinie (oben im Bild), y=68 die Kameraseite (unten).
    # Flip damit y=0 unten (Kameraseite) und y=68 oben (Ferne Seite) entspricht.
    df["y_m"]  = PITCH_WIDTH_M - y2d_out / SCALE_Y

    # Außerhalb des Feldes markieren (z.B. Kamerarand-Artefakte)
    in_bounds = (
        (df["x_m"] >= -2) & (df["x_m"] <= PITCH_LENGTH_M + 2) &
        (df["y_m"] >= -2) & (df["y_m"] <= PITCH_WIDTH_M  + 2)
    )
    df["in_bounds"] = in_bounds
    out_count = (~in_bounds).sum()
    if out_count > 0:
        print(f"  ⚠  {out_count} Punkte außerhalb des Feldes (in_bounds=False)")

    print(f"[Transform] Fertig.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. SPRUNGFILTERUNG & SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def filter_and_smooth_trajectories(
    df: pd.DataFrame,
    fps: float = 25.0,
    max_speed_m_s: float = 12.0,
    smooth_window: int = 9,
    smooth_poly: int = 2,
) -> pd.DataFrame:
    """
    Entfernt physikalisch unmögliche Sprünge in Trajektorien und glättet sie.

    Für jede track_id:
      1. Aufeinanderfolgende Frames werden verglichen. Wenn die Distanz
         größer als max_speed_m_s * frame_dt ist, wird die Trajektorie
         an dieser Stelle aufgeschnitten -> neues Segment beginnt.
      2. Innerhalb jedes Segments: Savitzky-Golay-Glättung auf x_m und y_m.

    Fügt Spalte 'segment_id' hinzu (eindeutig pro Track-Segment).
    Zeichnet visualize_trajectories() dann pro segment_id, entstehen
    keine Linien über Sprünge hinweg.

    Args:
        fps:            Framerate des Videos (für Geschwindigkeitsberechnung)
        max_speed_m_s:  Max. erlaubte Spielergeschwindigkeit (12 m/s ≈ 43 km/h)
        smooth_window:  Fenstergröße für Savitzky-Golay (muss ungerade sein)
        smooth_poly:    Polynomgrad für Savitzky-Golay
    """
    if "track_id" not in df.columns:
        print("[Smoothing] Keine track_id Spalte – übersprungen.")
        df["segment_id"] = 0
        return df

    df = df.copy()
    df["segment_id"] = -1

    seg_counter = 0
    jump_count = 0
    short_segs = 0

    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        idx = group.index
        frames = group["frame_idx"].values
        xs = group["x_m"].values.copy()
        ys = group["y_m"].values.copy()

        # Sprungpositionen bestimmen (Grenze zwischen zwei Segmenten)
        seg_boundaries = [0]
        for i in range(1, len(frames)):
            dt = (int(frames[i]) - int(frames[i - 1])) / fps
            dist = float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
            if dist > max_speed_m_s * dt:
                seg_boundaries.append(i)
                jump_count += 1
        seg_boundaries.append(len(frames))

        # Pro Segment: segment_id vergeben + glätten
        for s in range(len(seg_boundaries) - 1):
            a, b = seg_boundaries[s], seg_boundaries[s + 1]
            seg_idx = idx[a:b]
            df.loc[seg_idx, "segment_id"] = seg_counter

            seg_len = b - a
            if seg_len >= smooth_window:
                df.loc[seg_idx, "x_m"] = savgol_filter(xs[a:b], smooth_window, smooth_poly)
                df.loc[seg_idx, "y_m"] = savgol_filter(ys[a:b], smooth_window, smooth_poly)
            else:
                short_segs += 1

            seg_counter += 1

    print(f"[Smoothing] {jump_count} Sprünge entfernt -> {seg_counter} Segmente "
          f"({short_segs} zu kurz zum Glätten, window={smooth_window})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISIERUNG
# ─────────────────────────────────────────────────────────────────────────────

def draw_pitch(ax, length=PITCH_LENGTH_M, width=PITCH_WIDTH_M):
    """Zeichnet einen Fußballplatz (Top-View) in Metern."""
    ax.set_facecolor("#3a7d44")

    def line(x1, y1, x2, y2, **kw):
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=1.5, **kw)

    # Außenlinien
    line(0, 0, length, 0); line(0, width, length, width)
    line(0, 0, 0, width);  line(length, 0, length, width)

    # Mittellinie
    line(length/2, 0, length/2, width)

    # Mittelkreis
    circle = plt.Circle((length/2, width/2), 9.15, color="white", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.plot(length/2, width/2, 'o', color="white", markersize=3)

    # Strafräume
    for x0 in [0, length - 16.5]:
        rect = patches.Rectangle((x0, 13.84), 16.5, 40.32,
                                  linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)

    # Toräume
    for x0 in [0, length - 5.5]:
        rect = patches.Rectangle((x0, 24.84), 5.5, 18.32,
                                  linewidth=1.5, edgecolor="white", facecolor="none")
        ax.add_patch(rect)

    # Elfmeterpunkte
    ax.plot(11, width/2, 'o', color="white", markersize=3)
    ax.plot(length - 11, width/2, 'o', color="white", markersize=3)

    ax.set_xlim(-2, length + 2)
    ax.set_ylim(-2, width + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Länge (m)")
    ax.set_ylabel("Breite (m)")


def visualize_frame(df: pd.DataFrame, frame_idx: int, output_path: str = None):
    """Zeigt die Top-View Positionen für einen einzelnen Frame."""
    frame_df = df[df["frame_idx"] == frame_idx]
    if frame_df.empty:
        print(f"Keine Daten für Frame {frame_idx}")
        return

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax)

    # Farben: Real Madrid Spieler rot, Malaga blau, Ball gelb
    REAL_MADRID_PLAYERS = {
        "danilo", "varane", "ramos", "marcelo", "modric",
        "casemiro", "kroos", "isco", "benzema", "ronaldo"
    }
    colors = {"real_madrid": "#e63946", "opponent": "#457b9d", "ball": "#f4d03f"}

    if "object_type" in frame_df.columns and "clean_class_name" in frame_df.columns:
        # Ball separat
        ball = frame_df[(frame_df["object_type"] == "ball") & frame_df["in_bounds"]]
        if not ball.empty:
            ax.scatter(ball["x_m"], ball["y_m"],
                       c="#f4d03f", s=200, marker="*",
                       label="ball", zorder=6, edgecolors="black", linewidths=0.5)

        # Spieler mit Namen beschriften
        players = frame_df[(frame_df["object_type"] == "player") & frame_df["in_bounds"]]
        for _, row in players.iterrows():
            name = str(row.get("clean_class_name", "")).lower()
            color = "#e63946" if name in REAL_MADRID_PLAYERS else "#457b9d"
            ax.scatter(row["x_m"], row["y_m"],
                       c=color, s=100, marker="o",
                       zorder=5, edgecolors="white", linewidths=0.8)
            ax.text(row["x_m"] + 0.5, row["y_m"] + 0.5,
                    name, fontsize=6, color="white", zorder=7)

        # Legende manuell
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e63946",
                   markersize=8, label="Real Madrid"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#457b9d",
                   markersize=8, label="Malaga"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#f4d03f",
                   markersize=10, label="Ball"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.7)
    elif "team" in frame_df.columns:
        for team, group in frame_df.groupby("team"):
            color  = colors.get(team, "white")
            size   = 150 if team == "ball" else 80
            marker = "*" if team == "ball" else "o"
            valid  = group[group["in_bounds"]]
            ax.scatter(valid["x_m"], valid["y_m"],
                       c=color, s=size, marker=marker,
                       label=team, zorder=5, edgecolors="white", linewidths=0.5)
        ax.legend(loc="upper right", framealpha=0.7)
    else:
        valid = frame_df[frame_df["in_bounds"]]
        ax.scatter(valid["x_m"], valid["y_m"], c="white", s=80, zorder=5)

    ax.set_title(f"Top-View – Frame {frame_idx}", fontsize=13, fontweight="bold", color="white")
    ax.legend(loc="upper right", framealpha=0.7)
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Visualisierung] Gespeichert: {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_trajectories(df: pd.DataFrame, track_id: int = None, output_path: str = None):
    """Zeichnet Trajektorien über den gesamten Clip."""
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax)

    valid = df[df["in_bounds"]].copy()
    valid = valid.sort_values("frame_idx")

    if track_id is not None:
        valid = valid[valid["track_id"] == track_id]
        ax.plot(valid["x_m"], valid["y_m"], linewidth=1.5, color="yellow", alpha=0.8)
        ax.scatter(valid["x_m"].iloc[0], valid["y_m"].iloc[0],
                   c="green", s=100, zorder=6, label="Start")
        ax.scatter(valid["x_m"].iloc[-1], valid["y_m"].iloc[-1],
                   c="red", s=100, zorder=6, label="Ende")
        ax.set_title(f"Trajektorie – Track ID {track_id}", fontsize=13, color="white")
    else:
        # Alle Tracks – pro Segment zeichnen (keine Linien über Sprünge)
        if "track_id" in valid.columns:
            n_tracks = valid["track_id"].nunique()
            cmap = plt.cm.get_cmap("tab20", n_tracks)
            track_ids = sorted(valid["track_id"].unique())
            color_map = {tid: cmap(i) for i, tid in enumerate(track_ids)}

            group_col = "segment_id" if "segment_id" in valid.columns else "track_id"
            for (tid, seg), group in valid.groupby(["track_id", group_col]):
                group = group.sort_values("frame_idx")
                ax.plot(group["x_m"], group["y_m"],
                        linewidth=0.8, alpha=0.6, color=color_map[tid])
        else:
            ax.plot(valid["x_m"], valid["y_m"], linewidth=0.5, alpha=0.4, color="white")
        ax.set_title("Alle Trajektorien", fontsize=13, color="white")

    ax.legend(loc="upper right", framealpha=0.7)
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Visualisierung] Gespeichert: {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_player_trajectories(df: pd.DataFrame, output_path: str = None,
                                   fps: float = 25.0):
    """
    Zeichnet für jeden benannten Spieler eine eigene Trajektorie.
    Farbe = Zeit (blau = früh, gelb/rot = spät).
    """
    from matplotlib.collections import LineCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    REAL_MADRID_PLAYERS = {
        "danilo", "varane", "ramos", "marcelo", "modric",
        "casemiro", "kroos", "isco", "benzema", "ronaldo"
    }

    valid = df[
        (df["in_bounds"]) &
        (df["object_type"] == "player") &
        (df["clean_class_name"].notna())
    ].copy()

    named = valid[~valid["clean_class_name"].str.lower().isin(["opponent", "player", ""])]
    players = sorted(named["clean_class_name"].str.lower().unique())

    if not players:
        print("[Trajektorien] Keine benannten Spieler gefunden.")
        return

    # Globaler Zeitbereich für einheitliche Farbskala
    t_min = int(df["frame_idx"].min())
    t_max = int(df["frame_idx"].max())
    norm = Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.cm.plasma

    n = len(players)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    fig.patch.set_facecolor("#1a1a2e")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, name in enumerate(players):
        ax = axes_flat[i]
        draw_pitch(ax)

        pdata = named[named["clean_class_name"].str.lower() == name].sort_values("frame_idx")

        # Trajektorie als farbige Liniensegmente — nur bei kleinen Frame-Lücken
        max_gap = fps * 1.5  # max. 1.5 Sekunden Lücke
        if len(pdata) >= 2:
            xs = pdata["x_m"].values
            ys = pdata["y_m"].values
            ts = pdata["frame_idx"].values.astype(float)

            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            all_segs = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_t    = (ts[:-1] + ts[1:]) / 2
            gaps     = np.diff(ts)

            # Nur Segmente ohne große Lücke zeichnen
            valid_mask = gaps <= max_gap
            if valid_mask.any():
                lc = LineCollection(all_segs[valid_mask], cmap=cmap, norm=norm,
                                    linewidth=2, alpha=0.9)
                lc.set_array(seg_t[valid_mask])
                ax.add_collection(lc)

            # Start- und Endpunkt markieren
            ax.scatter(xs[0],  ys[0],  c="lime",  s=60, zorder=6, marker="o")
            ax.scatter(xs[-1], ys[-1], c="red",   s=60, zorder=6, marker="X")

        team = "Real Madrid" if name in REAL_MADRID_PLAYERS else "Malaga"
        ax.set_title(f"{name.capitalize()} ({team})", fontsize=9,
                     color="white", fontweight="bold")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.set_xlabel("Länge (m)", fontsize=7)
        ax.set_ylabel("Breite (m)", fontsize=7)

    # Leere Subplots ausblenden
    for j in range(len(players), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Spieler-Trajektorien (Farbe = Zeit)", fontsize=13,
                 color="white", fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 0.95, 0.98])

    # Colorbar ganz rechts außerhalb der Subplots
    cbar_ax = fig.add_axes([0.96, 0.1, 0.015, 0.8])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label="Zeit")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    tick_frames = np.linspace(t_min, t_max, 6).astype(int)
    tick_labels = [f"{int(f/fps)//60:02d}:{int(f/fps)%60:02d}" for f in tick_frames]
    cbar.set_ticks(tick_frames)
    cbar.set_ticklabels(tick_labels)

    # Legende: Startpunkt / Endpunkt unten zentriert
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", markersize=7, label="Start", linestyle="None"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red",  markersize=7, label="Ende",  linestyle="None"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               framealpha=0.4, labelcolor="white", fontsize=9, bbox_to_anchor=(0.47, 0.0))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Visualisierung] Gespeichert: {output_path}")
    else:
        plt.show()
    plt.close()


def visualize_heatmap(df: pd.DataFrame, output_path: str = None,
                      object_type: str = "player", title: str = None,
                      bins: int = 50):
    """Heatmap der Aufenthaltswahrscheinlichkeit auf dem Spielfeld."""
    from scipy.ndimage import gaussian_filter

    valid = df[(df["in_bounds"]) & (df["object_type"] == object_type)].copy()

    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax)

    # 2D-Histogramm
    H, xedges, yedges = np.histogram2d(
        valid["x_m"], valid["y_m"],
        bins=bins,
        range=[[0, PITCH_LENGTH_M], [0, PITCH_WIDTH_M]],
    )
    H = gaussian_filter(H.T, sigma=1.5)

    im = ax.imshow(
        H,
        origin="lower",
        extent=[0, PITCH_LENGTH_M, 0, PITCH_WIDTH_M],
        cmap="hot",
        alpha=0.65,
        aspect="auto",
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax, label="Aufenthalts-Frames", pad=0.01)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    label = "Spieler" if object_type == "player" else "Ball"
    ax.set_title(title or f"Heatmap – {label}", fontsize=13, color="white")
    ax.set_xlabel("Länge (m)", color="white")
    ax.set_ylabel("Breite (m)", color="white")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Visualisierung] Gespeichert: {output_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Homography Pipeline: Video zu 2D Top-View Tracking")
    parser.add_argument("--video",           type=str, default=None,         help="Pfad zum Video (.mp4)")
    parser.add_argument("--parquet",         type=str, required=True,        help="Pfad zur Tracking-Parquet-Datei")
    parser.add_argument("--output_dir",      type=str, default=f"{TRACKING_DIR}/homography", help="Ausgabeverzeichnis")
    parser.add_argument("--sample_every",    type=int, default=5,            help="Jeden N-ten Frame fuer Kalibrierung verwenden")
    parser.add_argument("--homographies",    type=str, default=None,         help="Vorberechnete Homographien (.pkl) laden/speichern")
    parser.add_argument("--visualize_only",  action="store_true",            help="Nur Visualisierung (Parquet muss x_2d enthalten)")
    parser.add_argument("--visualize_frame", type=int, default=None,         help="Frame-Index fuer Einzelframe-Visualisierung")
    parser.add_argument("--visualize_track", type=int, default=None,         help="Track-ID fuer Trajektorie")
    parser.add_argument("--fps",             type=float, default=25.0,       help="Framerate des Videos (fuer Sprungfilter)")
    parser.add_argument("--max_speed",       type=float, default=12.0,       help="Max. Spielergeschwindigkeit in m/s (Standard: 12.0)")
    parser.add_argument("--smooth_window",   type=int,   default=9,          help="Savitzky-Golay Fenstergroesse (ungerade, Standard: 9)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Nur Visualisierung ──────────────────────────────────────────────────
    if args.visualize_only:
        print("[Modus] Nur Visualisierung")
        df = pd.read_parquet(args.parquet)
        if "x_2d" not in df.columns:
            print("FEHLER: Parquet enthält keine 'x_2d' Spalte. Pipeline zuerst vollständig ausführen.")
            sys.exit(1)
        if "in_bounds" not in df.columns:
            df["in_bounds"] = True

        if args.visualize_frame is not None:
            visualize_frame(df, args.visualize_frame,
                            str(output_dir / f"frame_{args.visualize_frame}_topview.png"))
        else:
            visualize_trajectories(df, args.visualize_track,
                                   str(output_dir / "trajectories.png"))
        return

    # ── Vollständige Pipeline ───────────────────────────────────────────────
    if args.video is None:
        print("FEHLER: --video ist erforderlich (außer bei --visualize_only)")
        sys.exit(1)

    print("=" * 60)
    print("  HOMOGRAPHY PIPELINE")
    print("=" * 60)

    # 1. Tracking laden
    print(f"\n[1/4] Lade Tracking-Daten: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"      {len(df)} Zeilen, Spalten: {list(df.columns)}")
    # Nur frame umbenennen — x/y werden in transform_tracking gesetzt
    df = df.rename(columns={"frame": "frame_idx"})

    total_frames = int(df["frame_idx"].max()) + 1

    # 2. Frames extrahieren
    frames_dir = str(output_dir / "frames")
    if not Path(frames_dir).exists() or len(list(Path(frames_dir).glob("*.jpg"))) == 0:
        print(f"\n[2/4] Frame-Extraktion")
        extract_frames(args.video, frames_dir, args.sample_every)
    else:
        n = len(list(Path(frames_dir).glob("*.jpg")))
        print(f"\n[2/4] Frames bereits vorhanden ({n} Frames in '{frames_dir}')")

    # 3. Homographien berechnen oder laden
    pkl_path = args.homographies or str(output_dir / "homographies.pkl")
    if Path(pkl_path).exists():
        print(f"\n[3/4] Homographien laden: {pkl_path}")
        with open(pkl_path, "rb") as f:
            homographies = pickle.load(f)
        print(f"      {len(homographies)} Homographien geladen")
    else:
        print(f"\n[3/4] Homographien berechnen (PnLCalib)")
        homographies = run_pnlcalib(frames_dir, pkl_path)

    # Kamerawechsel-Filter
    homographies = filter_homography_valid(homographies)

    # Interpolator erstellen
    get_H = build_interpolator(homographies, total_frames)

    # 4. Tracking transformieren
    print(f"\n[4/4] Tracking transformieren")
    df_2d = transform_tracking(df, get_H, total_frames)

    # Sprungfilterung & Smoothing
    print(f"\n[5/5] Sprungfilterung & Smoothing")
    df_2d = filter_and_smooth_trajectories(
        df_2d,
        fps=args.fps,
        max_speed_m_s=args.max_speed,
        smooth_window=args.smooth_window,
        smooth_poly=2,
    )

    # Speichern
    out_parquet = str(output_dir / "tracking_2d.parquet")
    df_2d.to_parquet(out_parquet, index=False)
    print(f"      Gespeichert: {out_parquet}")

    # Statistiken
    print(f"\n── Statistiken ──────────────────────────────────────")
    print(f"  Frames abgedeckt:  {df_2d['frame_idx'].nunique()}")
    print(f"  x_m Bereich:       {df_2d['x_m'].min():.1f} – {df_2d['x_m'].max():.1f} m")
    print(f"  y_m Bereich:       {df_2d['y_m'].min():.1f} – {df_2d['y_m'].max():.1f} m")
    if "in_bounds" in df_2d.columns:
        pct = df_2d["in_bounds"].mean() * 100
        print(f"  In-Bounds:         {pct:.1f}%")
    print(f"─────────────────────────────────────────────────────\n")

    # Visualisierungen
    print("[Visualisierung] Erstelle Plots...")
    mid_frame = df_2d["frame_idx"].median()
    sample_frame = df_2d.iloc[(df_2d["frame_idx"] - mid_frame).abs().argsort()].iloc[0]["frame_idx"]
    visualize_frame(df_2d, int(sample_frame),
                    str(output_dir / "sample_frame_topview.png"))
    visualize_trajectories(df_2d, args.visualize_track,
                           str(output_dir / "trajectories.png"))

    print("\n Pipeline abgeschlossen!")
    print(f"   Ergebnisse in: {output_dir}/")
    print(f"   tracking_2d.parquet  (Tracking + x_2d, y_2d, x_m, y_m)")
    print(f"   homographies.pkl     (H-Matrizen pro Frame)")
    print(f"   sample_frame_topview.png")
    print(f"   trajectories.png")


if __name__ == "__main__":
    main()