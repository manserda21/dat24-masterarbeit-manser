from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# -----------------------------
# Pitch drawing utilities
# -----------------------------
@dataclass(frozen=True)
class PitchSpec:
    # Standard soccer pitch in meters (can be adjusted)
    length: float = 105.0
    width: float = 68.0

    # Visual style (tune these)
    grass: str = "#1E5B3A"      # darker green
    grass_alt: str = "#1A5033"  # subtle stripe
    line: str = "#FFFFFF"       # field lines
    line_alpha: float = 0.95

    # Line widths
    lw_outer: float = 2.2
    lw_inner: float = 1.6


def draw_pitch(ax: plt.Axes, spec: PitchSpec, stripe_count: int = 10) -> None:
    """
    Draw a top-down pitch from (0,0) bottom-left to (L,W) top-right.
    """
    L, W = spec.length, spec.width

    # Background
    ax.set_facecolor(spec.grass)

    # Subtle vertical stripes
    stripe_w = L / stripe_count
    for i in range(stripe_count):
        if i % 2 == 0:
            ax.add_patch(
                plt.Rectangle((i * stripe_w, 0), stripe_w, W,
                              facecolor=spec.grass_alt, edgecolor="none", alpha=0.35)
            )

    # Outer boundary
    ax.add_patch(
        plt.Rectangle((0, 0), L, W, fill=False, edgecolor=spec.line,
                      linewidth=spec.lw_outer, alpha=spec.line_alpha)
    )

    # Halfway line
    ax.plot([L / 2, L / 2], [0, W], color=spec.line, lw=spec.lw_inner, alpha=spec.line_alpha)

    # Center circle + spot
    center = (L / 2, W / 2)
    ax.add_patch(
        plt.Circle(center, 9.15, fill=False, edgecolor=spec.line,
                   linewidth=spec.lw_inner, alpha=spec.line_alpha)
    )
    ax.scatter([center[0]], [center[1]], s=18, c=spec.line, alpha=spec.line_alpha)

    # Penalty areas and 6-yard boxes (dimensions in meters)
    # Left
    ax.add_patch(plt.Rectangle((0, (W - 40.32) / 2), 16.5, 40.32,
                               fill=False, edgecolor=spec.line, lw=spec.lw_inner, alpha=spec.line_alpha))
    ax.add_patch(plt.Rectangle((0, (W - 18.32) / 2), 5.5, 18.32,
                               fill=False, edgecolor=spec.line, lw=spec.lw_inner, alpha=spec.line_alpha))
    ax.scatter([11.0], [W / 2], s=18, c=spec.line, alpha=spec.line_alpha)

    # Right
    ax.add_patch(plt.Rectangle((L - 16.5, (W - 40.32) / 2), 16.5, 40.32,
                               fill=False, edgecolor=spec.line, lw=spec.lw_inner, alpha=spec.line_alpha))
    ax.add_patch(plt.Rectangle((L - 5.5, (W - 18.32) / 2), 5.5, 18.32,
                               fill=False, edgecolor=spec.line, lw=spec.lw_inner, alpha=spec.line_alpha))
    ax.scatter([L - 11.0], [W / 2], s=18, c=spec.line, alpha=spec.line_alpha)

    # Arc (approximate) - optional, keep simple for poster
    # You can add penalty arcs later if you want.

    # Axis formatting
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# -----------------------------
# Synthetic data generators
# -----------------------------
def generate_synthetic_trajectories(
    n_players: int = 10,
    n_frames: int = 300,
    pitch: PitchSpec = PitchSpec(),
    seed: int = 7,
) -> Dict[str, np.ndarray]:
    """
    Returns dict: player_id -> array shape (n_frames, 2) with (x,y).
    Smooth-ish random walks for a nice illustrative figure.
    """
    rng = np.random.default_rng(seed)
    L, W = pitch.length, pitch.width

    traj = {}
    for i in range(n_players):
        x0 = rng.uniform(10, L - 10)
        y0 = rng.uniform(8, W - 8)

        # velocity noise (smoothed)
        vx = rng.normal(0, 0.6, size=n_frames)
        vy = rng.normal(0, 0.6, size=n_frames)
        vx = np.convolve(vx, np.ones(15) / 15.0, mode="same")
        vy = np.convolve(vy, np.ones(15) / 15.0, mode="same")

        x = x0 + np.cumsum(vx)
        y = y0 + np.cumsum(vy)

        # keep inside pitch with padding
        x = np.clip(x, 1.0, L - 1.0)
        y = np.clip(y, 1.0, W - 1.0)

        traj[f"P{i+1}"] = np.column_stack([x, y])

    return traj


def generate_synthetic_passing_network(
    players: List[str],
    seed: int = 11,
) -> List[Tuple[str, str, int]]:
    """
    Returns list of directed edges (u,v,weight).
    """
    rng = np.random.default_rng(seed)
    edges = []
    n = len(players)

    # Create a sparse-ish directed weighted network
    for _ in range(n * 2):
        u = players[rng.integers(0, n)]
        v = players[rng.integers(0, n)]
        if u == v:
            continue
        w = int(rng.integers(2, 18))
        edges.append((u, v, w))

    # Merge duplicates
    merged: Dict[Tuple[str, str], int] = {}
    for u, v, w in edges:
        merged[(u, v)] = merged.get((u, v), 0) + w

    return [(u, v, w) for (u, v), w in merged.items()]


# -----------------------------
# Plotting: trajectories
# -----------------------------
def plot_trajectories(
    trajectories: Dict[str, np.ndarray],
    outpath: Path,
    pitch: PitchSpec = PitchSpec(),
    title: str = "Reconstructed trajectories (illustrative)",
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=200)
    draw_pitch(ax, pitch)

    # Use a matplotlib colormap (no manual color picking needed)
    cmap = plt.get_cmap("tab10")
    for idx, (pid, xy) in enumerate(trajectories.items()):
        color = cmap(idx % 10)

        # line
        ax.plot(
            xy[:, 0], xy[:, 1],
            lw=2.0,
            alpha=0.70,
            color=color,
        )

        # small marker at last position
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], s=30, color=color, edgecolors="none", alpha=0.95)

    ax.set_title(title, fontsize=16, pad=10)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting: passing network
# -----------------------------
def plot_passing_network(
    trajectories: Dict[str, np.ndarray],
    edges: List[Tuple[str, str, int]],
    outpath: Path,
    pitch: PitchSpec = PitchSpec(),
    title: str = "Passing network (illustrative)",
) -> None:
    # Node positions from mean location over time (or use first/last, depending on your story)
    pos: Dict[str, Tuple[float, float]] = {}
    for pid, xy in trajectories.items():
        pos[pid] = (float(xy[:, 0].mean()), float(xy[:, 1].mean()))

    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    # Scale edge widths
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    if len(weights) == 0:
        w_scaled = []
    else:
        w_min, w_max = float(weights.min()), float(weights.max())
        # Avoid division by zero
        denom = (w_max - w_min) if (w_max > w_min) else 1.0
        w_scaled = 1.0 + 6.0 * ((weights - w_min) / denom)

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=200)
    draw_pitch(ax, pitch)

    # Draw edges manually for better control (arrows + alpha)
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        lw = float(w_scaled[i]) if i < len(w_scaled) else 2.0

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=lw,
                shrinkA=10,
                shrinkB=10,
                alpha=0.55,
                color="white",
            ),
        )

    # Nodes: white fill + green edge (matches your theme)
    xs = [pos[p][0] for p in pos]
    ys = [pos[p][1] for p in pos]
    ax.scatter(xs, ys, s=220, facecolors="white", edgecolors=pitch.grass_alt, linewidths=2.0, zorder=5)

    # Labels
    for p, (x, y) in pos.items():
        ax.text(x, y, p, ha="center", va="center", fontsize=10, color="black", zorder=6)

    ax.set_title(title, fontsize=16, pad=10)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pitch = PitchSpec(
        grass="#1E5B3A",
        grass_alt="#184A30",
        line="#FFFFFF",
        line_alpha=0.95,
        lw_outer=2.2,
        lw_inner=1.6,
    )

    trajectories = generate_synthetic_trajectories(n_players=10, n_frames=320, pitch=pitch, seed=7)
    players = list(trajectories.keys())
    edges = generate_synthetic_passing_network(players, seed=11)

    plot_trajectories(
        trajectories,
        outpath=Path("images") / "figure_trajectories.pdf",
        pitch=pitch,
        title="Reconstructed trajectories (illustrative)",
    )

    plot_passing_network(
        trajectories,
        edges,
        outpath=Path("images") / "figure_passing_network_2.pdf",
        pitch=pitch,
        title="Passing network (illustrative)",
    )

    print("Saved:")
    print(" - images/figure_trajectories.pdf")
    print(" - images/figure_passing_network_2.pdf")


if __name__ == "__main__":
    main()