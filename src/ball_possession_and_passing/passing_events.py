"""
Passing Event Detection
=======================
Erkennt Pässe aus 2D-Trajektorien (tracking_2d.parquet).

Algorithmus:
  1. Ball-Besitz: Spieler mit geringstem Abstand zum Ball < possession_radius_m
  2. Pass erkannt wenn Besitzer wechselt und entweder:
       a) Ball war mind. min_flight_frames Frames ohne Besitzer (Flugphase), ODER
       b) maximale Ballgeschwindigkeit während des Wechsels > ball_speed_min_m_s

Verwendung:
    python passing_events.py \\
        --parquet /data/manser/homography_outputs/.../tracking_2d.parquet \\
        --output_dir /data/manser/passing_outputs/v1
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0


# ─────────────────────────────────────────────────────────────────────────────
# 0. BALL-INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_ball_gaps(ball: pd.DataFrame, max_gap_frames: int = 30) -> pd.DataFrame:
    """
    Interpoliert kurze Lücken in der Ball-Trajektorie linear.

    Wenn der Ball für ≤ max_gap_frames Frames nicht detektiert wird
    (z.B. kurz verdeckt), werden die fehlenden Frames durch lineare
    Interpolation zwischen letztem und erstem bekannten Punkt ergänzt.

    Hohe Pässe (lange Lücken > max_gap_frames) werden NICHT interpoliert,
    da der Ball dort einen Bogen fliegt — lineare Interpolation wäre falsch.
    """
    if ball.empty:
        return ball

    frames = ball["frame_idx"].values
    all_frames = np.arange(frames.min(), frames.max() + 1)
    missing = np.setdiff1d(all_frames, frames)

    if len(missing) == 0:
        return ball

    # Nur Lücken ≤ max_gap_frames interpolieren
    gaps = np.split(missing, np.where(np.diff(missing) > 1)[0] + 1)
    interpolated_rows = []

    for gap in gaps:
        if len(gap) > max_gap_frames:
            continue

        f_before = gap[0] - 1
        f_after  = gap[-1] + 1

        row_before = ball[ball["frame_idx"] == f_before]
        row_after  = ball[ball["frame_idx"] == f_after]

        if row_before.empty or row_after.empty:
            continue

        x0, y0 = row_before["x_m"].values[0], row_before["y_m"].values[0]
        x1, y1 = row_after["x_m"].values[0],  row_after["y_m"].values[0]
        n = len(gap) + 1

        for i, fi in enumerate(gap, start=1):
            t = i / n
            interpolated_rows.append({
                "frame_idx":   int(fi),
                "x_m":         x0 + t * (x1 - x0),
                "y_m":         y0 + t * (y1 - y0),
                "object_type": "ball",
                "in_bounds":   True,
                "interpolated": True,
            })

    if not interpolated_rows:
        ball["interpolated"] = False
        return ball

    ball = ball.copy()
    ball["interpolated"] = False
    df_new = pd.DataFrame(interpolated_rows)
    ball = pd.concat([ball, df_new], ignore_index=True).sort_values("frame_idx").reset_index(drop=True)

    print(f"        Ball-Interpolation: {len(interpolated_rows)} Frames ergänzt "
          f"(max_gap={max_gap_frames})")
    return ball


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATEN LADEN
# ─────────────────────────────────────────────────────────────────────────────

def load_data(parquet_path: str) -> tuple:
    """Lädt tracking_2d.parquet und trennt Ball- und Spieler-Daten."""
    df = pd.read_parquet(parquet_path)
    df = df.rename(columns={"frame": "frame_idx"})

    required = {"frame_idx", "x_m", "y_m", "object_type", "in_bounds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")

    ball    = df[(df["object_type"] == "ball") & df["in_bounds"]].copy()
    players = df[(df["object_type"] == "player") & df["in_bounds"]].copy()

    ball = ball.sort_values("frame_idx").reset_index(drop=True)

    # Kurze Lücken in der Ball-Trajektorie linear interpolieren (max. 30 Frames)
    ball = interpolate_ball_gaps(ball, max_gap_frames=30)

    # Positionsausreisser entfernen: wenn Ball in einem Frame >30m springt
    # → wahrscheinlich falsche Detektion (kein echter Ball)
    ball["_dx"] = ball["x_m"].diff().abs()
    ball["_dy"] = ball["y_m"].diff().abs()
    ball["_jump"] = np.hypot(ball["_dx"], ball["_dy"])
    # Ein Frame ist Ausreisser wenn er sowohl einen großen Sprung hat UND
    # der nächste Frame ebenfalls einen großen Sprung zeigt (d.h. der Frame
    # ist isoliert, nicht der Ball selbst der sich bewegt)
    outlier = (ball["_jump"] > 30) & (ball["_jump"].shift(-1) > 30)
    n_outlier = outlier.sum()
    if n_outlier > 0:
        print(f"        Ball-Ausreisser entfernt: {n_outlier} Frames")
        ball = ball[~outlier].copy()
    ball = ball.drop(columns=["_dx", "_dy", "_jump"]).reset_index(drop=True)

    fps = 25.0
    if "fps" in df.columns:
        fps = float(df["fps"].dropna().iloc[0]) if not df["fps"].dropna().empty else 25.0

    print(f"[Daten] {df['frame_idx'].nunique()} Frames, fps={fps}")
    print(f"        Ball:    {len(ball)} Detektionen")
    print(f"        Spieler: {len(players)} Detektionen, "
          f"{players['track_id'].nunique()} Tracks")

    return ball, players, fps


# ─────────────────────────────────────────────────────────────────────────────
# 2. BALLBESITZ ZUWEISEN
# ─────────────────────────────────────────────────────────────────────────────

def assign_possession(ball: pd.DataFrame,
                      players: pd.DataFrame,
                      fps: float = 25.0,
                      possession_radius_m: float = 2.5) -> pd.DataFrame:
    """
    Berechnet pro Frame: wer hat den Ball?

    Strategie:
      Nächster Spieler innerhalb possession_radius_m bekommt Besitz.
      Kein Ball-Speed-Filter: der Ball kann auch kurz nach einem hohen Pass
      noch schnell sein, wenn er beim Spieler landet. Ausreisser werden
      bereits in load_data() entfernt.

    Returns DataFrame mit Spalten:
        frame_idx, ball_x, ball_y, ball_speed,
        possessor_id, possessor_name, possessor_team, possession_dist
    """
    ball = ball.copy().sort_values("frame_idx")
    ball["ball_vx"] = ball["x_m"].diff() * fps
    ball["ball_vy"] = ball["y_m"].diff() * fps
    ball["ball_speed"] = np.hypot(ball["ball_vx"], ball["ball_vy"]).fillna(0.0)

    b = ball[["frame_idx", "x_m", "y_m", "ball_speed"]].rename(
        columns={"x_m": "ball_x", "y_m": "ball_y"}
    )
    p = players[["frame_idx", "track_id", "x_m", "y_m"]].copy()
    p["name"] = players["clean_class_name"] if "clean_class_name" in players.columns \
                else players["track_id"].astype(str)
    p["team"] = players["team"] if "team" in players.columns else None

    merged = b.merge(p, on="frame_idx", how="left")
    merged["dist"] = np.hypot(merged["x_m"] - merged["ball_x"],
                               merged["y_m"] - merged["ball_y"])

    idx_min = merged.groupby("frame_idx")["dist"].idxmin()
    closest = merged.loc[idx_min].copy()

    # Kein Besitz wenn Ball zu weit weg
    in_control = closest["dist"] < possession_radius_m

    closest["possessor_id"]   = np.where(in_control, closest["track_id"], None)
    closest["possessor_name"] = np.where(in_control, closest["name"],     None)
    closest["possessor_team"] = np.where(in_control, closest["team"],     None)

    result = closest[[
        "frame_idx", "ball_x", "ball_y", "ball_speed",
        "possessor_id", "possessor_name", "possessor_team", "dist"
    ]].rename(columns={"dist": "possession_dist"}).sort_values("frame_idx")

    n_possessed = result["possessor_id"].notna().sum()
    print(f"[Besitz] {n_possessed}/{len(result)} Frames mit Ballbesitz "
          f"({n_possessed/len(result)*100:.1f}%)")

    return result.reset_index(drop=True)


def build_possession_segments(possession_df: pd.DataFrame,
                               fps: float = 25.0,
                               min_frames: int = 5,
                               gap_tolerance: int = 3) -> pd.DataFrame:
    """
    Fasst aufeinanderfolgende Frames gleichen Besitzers zu Segmenten zusammen.

    gap_tolerance: bis zu N Frames ohne Besitz innerhalb derselben Possession
                   werden überbrückt (kurze Verdeckungen, Messrauschen).
    min_frames:    kürzere Segmente werden verworfen.

    Returns DataFrame mit Spalten:
        passer_track_id, passer_name, passer_team,
        start_frame, end_frame, duration_frames, duration_s,
        avg_ball_x, avg_ball_y
    """
    pos = possession_df.sort_values("frame_idx").reset_index(drop=True)

    segments = []
    cur_id = cur_name = cur_team = None
    seg_start = None
    last_frame = None
    ball_xs = []
    ball_ys = []

    def close_segment():
        dur = last_frame - seg_start + 1
        if dur >= min_frames:
            segments.append({
                "track_id":       cur_id,
                "player_name":    cur_name,
                "team":           cur_team,
                "start_frame":    seg_start,
                "end_frame":      last_frame,
                "duration_frames": dur,
                "duration_s":     round(dur / fps, 3),
                "avg_ball_x":     float(np.mean(ball_xs)),
                "avg_ball_y":     float(np.mean(ball_ys)),
            })

    for _, row in pos.iterrows():
        fi  = int(row["frame_idx"])
        pid = row["possessor_id"]
        bx  = float(row["ball_x"])
        by  = float(row["ball_y"])

        if pid is not None:
            pid = int(pid) if isinstance(pid, (float, np.floating)) else pid

            if cur_id is None:
                # Erstes Segment starten
                cur_id, cur_name, cur_team = pid, row["possessor_name"], row["possessor_team"]
                seg_start = fi
                ball_xs, ball_ys = [bx], [by]
            elif pid == cur_id and (last_frame is None or fi <= last_frame + gap_tolerance + 1):
                # Gleicher Spieler, innerhalb Gap-Toleranz → weiter
                ball_xs.append(bx); ball_ys.append(by)
            else:
                # Spielerwechsel oder zu große Lücke → altes Segment schließen
                close_segment()
                cur_id, cur_name, cur_team = pid, row["possessor_name"], row["possessor_team"]
                seg_start = fi
                ball_xs, ball_ys = [bx], [by]
            last_frame = fi

    if cur_id is not None:
        close_segment()

    df_seg = pd.DataFrame(segments)
    print(f"[Segmente] {len(df_seg)} Besitz-Segmente "
          f"(min_frames={min_frames}, gap_tolerance={gap_tolerance})")
    return df_seg


# ─────────────────────────────────────────────────────────────────────────────
# 3. PÄSSE ERKENNEN
# ─────────────────────────────────────────────────────────────────────────────

def detect_passes(segments_df: pd.DataFrame,
                  possession_df: pd.DataFrame,
                  fps: float = 25.0,
                  max_pass_duration_s: float = 5.0) -> pd.DataFrame:
    """
    Erkennt Pässe aus aufeinanderfolgenden Besitz-Segmenten.

    Ein Pass ist ein Übergang von Segment A → Segment B wobei:
      - Passer ≠ Empfänger
      - Zeitlücke zwischen Ende von A und Start von B ≤ max_pass_duration_s

    Die Ballposition am Ende von A (x_start/y_start) und am Anfang von B
    (x_end/y_end) werden aus dem possession_df gelesen.
    """
    if segments_df.empty:
        return pd.DataFrame()

    segs = segments_df.sort_values("start_frame").reset_index(drop=True)
    pos  = possession_df.set_index("frame_idx")

    passes = []
    max_gap_frames = int(max_pass_duration_s * fps)

    # Nur named Spieler-Segmente (keine Opponents)
    named_segs = segs[segs["player_name"] != "opponent"].reset_index(drop=True)

    for i in range(len(named_segs) - 1):
        a = named_segs.iloc[i]
        b = named_segs.iloc[i + 1]

        # Zeitlücke zwischen Ende von A und Start von B (egal was dazwischen war)
        gap = int(b["start_frame"]) - int(a["end_frame"])

        if gap > max_gap_frames:
            continue
        if a["track_id"] == b["track_id"]:
            continue

        # Ballposition: Ende von Segment A und Anfang von Segment B
        fi_end   = int(a["end_frame"])
        fi_start = int(b["start_frame"])

        x_start = float(pos.loc[fi_end,   "ball_x"]) if fi_end   in pos.index else a["avg_ball_x"]
        y_start = float(pos.loc[fi_end,   "ball_y"]) if fi_end   in pos.index else a["avg_ball_y"]
        x_end   = float(pos.loc[fi_start, "ball_x"]) if fi_start in pos.index else b["avg_ball_x"]
        y_end   = float(pos.loc[fi_start, "ball_y"]) if fi_start in pos.index else b["avg_ball_y"]

        # Max Ballgeschwindigkeit während der Flugphase
        flight_frames = pos.loc[
            pos.index.intersection(range(fi_end + 1, fi_start))
        ]
        max_speed = float(flight_frames["ball_speed"].max()) if not flight_frames.empty else 0.0

        passes.append({
            "frame_start":        fi_end,
            "frame_end":          fi_start,
            "flight_frames":      gap,
            "duration_s":         round(gap / fps, 3),
            "passer_track_id":    a["track_id"],
            "passer_name":        a["player_name"],
            "passer_team":        a["team"],
            "receiver_track_id":  b["track_id"],
            "receiver_name":      b["player_name"],
            "receiver_team":      b["team"],
            "x_start":            round(x_start, 2),
            "y_start":            round(y_start, 2),
            "x_end":              round(x_end, 2),
            "y_end":              round(y_end, 2),
            "distance_m":         round(float(np.hypot(x_end - x_start, y_end - y_start)), 2),
            "max_ball_speed_m_s": round(max_speed, 2),
        })

    df_passes = pd.DataFrame(passes)

    if df_passes.empty:
        print("[Pässe] Keine Pässe erkannt – Parameter anpassen?")
        return df_passes

    print(f"[Pässe] {len(df_passes)} Pässe erkannt")
    print(f"        Distanz: Ø {df_passes['distance_m'].mean():.1f}m, "
          f"max {df_passes['distance_m'].max():.1f}m")
    print(f"        Dauer:   Ø {df_passes['duration_s'].mean():.2f}s")

    return df_passes


# ─────────────────────────────────────────────────────────────────────────────
# 4. HAS_BALL SPALTE
# ─────────────────────────────────────────────────────────────────────────────

def add_has_ball_column(tracking_df: pd.DataFrame,
                        possession_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt eine boolean Spalte 'has_ball' zum Tracking-DataFrame hinzu.

    has_ball=True wenn für diesen Frame der Spieler (track_id) als Besitzer
    in possession_df eingetragen ist. Ball-Zeilen und Spieler ohne Besitz
    erhalten has_ball=False.
    """
    tracking_df = tracking_df.copy()

    # Besitzer pro Frame: frame_idx → possessor_id
    pos_map = (
        possession_df[possession_df["possessor_id"].notna()]
        .set_index("frame_idx")["possessor_id"]
        .astype(str)
    )

    def _has_ball(row):
        if row.get("object_type") != "player":
            return False
        fi  = row["frame_idx"]
        tid = str(row["track_id"])
        return pos_map.get(fi) == tid

    tracking_df["has_ball"] = tracking_df.apply(_has_ball, axis=1)
    return tracking_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISIERUNG
# ─────────────────────────────────────────────────────────────────────────────

def draw_pitch(ax):
    ax.set_facecolor("#3a7d44")
    def line(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color="white", linewidth=1.5)

    line(0, 0, PITCH_LENGTH_M, 0); line(0, PITCH_WIDTH_M, PITCH_LENGTH_M, PITCH_WIDTH_M)
    line(0, 0, 0, PITCH_WIDTH_M); line(PITCH_LENGTH_M, 0, PITCH_LENGTH_M, PITCH_WIDTH_M)
    line(PITCH_LENGTH_M / 2, 0, PITCH_LENGTH_M / 2, PITCH_WIDTH_M)

    circle = plt.Circle((PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), 9.15,
                         color="white", fill=False, linewidth=1.5)
    ax.add_patch(circle)

    for x0 in [0, PITCH_LENGTH_M - 16.5]:
        ax.add_patch(patches.Rectangle((x0, 13.84), 16.5, 40.32,
                     linewidth=1.5, edgecolor="white", facecolor="none"))
    for x0 in [0, PITCH_LENGTH_M - 5.5]:
        ax.add_patch(patches.Rectangle((x0, 24.84), 5.5, 18.32,
                     linewidth=1.5, edgecolor="white", facecolor="none"))

    ax.set_xlim(-2, PITCH_LENGTH_M + 2)
    ax.set_ylim(-2, PITCH_WIDTH_M + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Länge (m)")
    ax.set_ylabel("Breite (m)")


def visualize_passes(df_passes: pd.DataFrame, output_path: str = None,
                     title: str = "Erkannte Pässe"):
    """Zeichnet alle Pässe als Pfeile auf dem Spielfeld."""
    fig, ax = plt.subplots(figsize=(14, 9))
    draw_pitch(ax)

    if df_passes.empty:
        ax.set_title("Keine Pässe erkannt", color="white")
    else:
        # Farbe nach Team
        team_colors = {}
        teams = df_passes["passer_team"].dropna().unique()
        palette = ["#e63946", "#457b9d", "#f4d03f", "#2ecc71"]
        for i, t in enumerate(teams):
            team_colors[t] = palette[i % len(palette)]

        for _, row in df_passes.iterrows():
            color = team_colors.get(row["passer_team"], "white")
            dx = row["x_end"] - row["x_start"]
            dy = row["y_end"] - row["y_start"]
            ax.annotate(
                "", xy=(row["x_end"], row["y_end"]),
                xytext=(row["x_start"], row["y_start"]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

        # Legende
        from matplotlib.lines import Line2D
        legend = [Line2D([0], [0], color=c, lw=2, label=t)
                  for t, c in team_colors.items()]
        ax.legend(handles=legend, loc="upper right", framealpha=0.7)

        ax.set_title(f"{title}  (n={len(df_passes)})", fontsize=13, color="white")

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


def visualize_possession_timeline(possession_df: pd.DataFrame,
                                  output_path: str = None):
    """Zeigt Ballbesitz über Zeit als Balkendiagramm."""
    pos = possession_df.copy()
    pos["has_ball"] = pos["possessor_id"].notna()

    fig, ax = plt.subplots(figsize=(14, 3))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    frames = pos["frame_idx"].values
    colors = ["#e63946" if h else "#2c3e50" for h in pos["has_ball"]]

    ax.bar(frames, 1, width=1, color=colors, alpha=0.8)
    ax.set_xlim(frames[0], frames[-1])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Frame", color="white")
    ax.set_title("Ballbesitz-Timeline  (rot = Besitz, dunkel = Flugphase)",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_yticks([])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Visualisierung] Gespeichert: {output_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Passing Event Detection aus 2D-Tracking")
    parser.add_argument("--parquet",             type=str,   required=True)
    parser.add_argument("--output_dir",          type=str,   required=True)
    parser.add_argument("--possession_radius",   type=float, default=2.5,
                        help="Besitzradius in Metern (Standard: 2.5)")
    parser.add_argument("--min_possession_frames", type=int, default=5,
                        help="Mindest-Besitzdauer in Frames (Standard: 5)")
    parser.add_argument("--gap_tolerance",       type=int,   default=3,
                        help="Max. Frames Lücke innerhalb einer Possession (Standard: 3)")
    parser.add_argument("--max_pass_duration_s", type=float, default=5.0,
                        help="Max. Flugzeit für gültigen Pass in Sekunden (Standard: 5.0)")
    parser.add_argument("--fps",                 type=float, default=25.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  PASSING EVENT DETECTION")
    print("=" * 55)
    print(f"\n  possession_radius:     {args.possession_radius} m")
    print(f"  min_possession_frames: {args.min_possession_frames}")
    print(f"  gap_tolerance:         {args.gap_tolerance} frames")
    print(f"  max_pass_duration_s:   {args.max_pass_duration_s} s\n")

    ball, players, fps = load_data(args.parquet)
    fps = args.fps

    print("\n[1/4] Ballbesitz berechnen...")
    possession = assign_possession(ball, players, fps=fps,
                                   possession_radius_m=args.possession_radius)

    print("\n[2/4] Besitz-Segmente bilden...")
    segments = build_possession_segments(possession, fps=fps,
                                         min_frames=args.min_possession_frames,
                                         gap_tolerance=args.gap_tolerance)

    print("\n[3/4] Pässe erkennen...")
    passes = detect_passes(segments, possession, fps=fps,
                           max_pass_duration_s=args.max_pass_duration_s)

    print("\n[4/4] Speichern & Visualisieren...")

    # has_ball Spalte zum Tracking-Parquet hinzufügen
    tracking_df = pd.read_parquet(args.parquet)
    tracking_df = tracking_df.rename(columns={"frame": "frame_idx"})
    tracking_df = add_has_ball_column(tracking_df, possession)
    tracking_df.to_parquet(args.parquet, index=False)
    has_ball_count = tracking_df["has_ball"].sum()
    print(f"      has_ball Spalte hinzugefügt: {has_ball_count} Einträge mit has_ball=True")

    possession.to_parquet(str(output_dir / "possession.parquet"), index=False)
    segments.to_parquet(str(output_dir / "possession_segments.parquet"), index=False)
    segments.to_csv(str(output_dir / "possession_segments.csv"), index=False)
    print(f"      Gespeichert: possession.parquet, possession_segments.csv")

    if not passes.empty:
        passes.to_parquet(str(output_dir / "passes.parquet"), index=False)
        passes.to_csv(str(output_dir / "passes.csv"), index=False)
        print(f"      Gespeichert: passes.parquet, passes.csv")
        print("\n── Pass-Statistiken ─────────────────────────────────")
        print(passes[["passer_name", "receiver_name", "distance_m",
                       "duration_s", "max_ball_speed_m_s"]].to_string(index=False))
        print("─────────────────────────────────────────────────────\n")

    visualize_passes(passes, str(output_dir / "passes_map.png"))
    visualize_possession_timeline(possession, str(output_dir / "possession_timeline.png"))

    print(f"\n Pipeline abgeschlossen! Ergebnisse in: {output_dir}/")
    print(f"   passes.parquet / passes.csv")
    print(f"   possession_segments.parquet / .csv")
    print(f"   possession.parquet")
    print(f"   passes_map.png  /  possession_timeline.png")


if __name__ == "__main__":
    main()
