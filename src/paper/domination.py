from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

from evaluation.domination import PairStats, Report, RobotReport
from paper.labels import METHODS, ROBOT_LABEL

# a count is a magnitude: one hue light -> dark, so no cell reads as a
# category (P. Wojdas's domination-comparison design)
RAMP = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
CMAP.set_bad("white")
INK_MUTED = "#52514e"
DIAGONAL = "#9a9893"
PANEL = 2.4


def _label(method: str) -> str:
    return METHODS.get(method, method)


def _fractions(pair: PairStats) -> np.ndarray:
    counts = np.asarray(pair.counts, dtype=float)
    return counts / counts.sum()


def shared_norm(pairs: list[PairStats]) -> LogNorm:
    """One log scale for every robot's heatmap, from a single level of the
    largest set to all levels."""
    levels = max(int(np.sum(p.counts)) for p in pairs)
    return LogNorm(vmin=1 / levels, vmax=1.0)


def stats_text(pair: PairStats) -> str:
    a, b = _label(pair.a), _label(pair.b)
    low, high = pair.mean_diff_ci
    return (
        f"{a} > {b} {pair.a_better:.0%} · {b} > {a} {pair.b_better:.0%}\n"
        f"tie {pair.ties:.0%} · Δ {pair.mean_diff:+.3f} "
        f"[{low:+.3f}, {high:+.3f}]\n"
        f"|Δ| {pair.mean_abs_diff:.3f}, noise {pair.noise_abs_diff:.3f}"
    )


def draw_pair(ax: Any, pair: PairStats, norm: LogNorm, title: str) -> Any:
    fractions = _fractions(pair)
    # histogram rows are A's bins (x); imshow rows are y
    image = ax.imshow(
        np.ma.masked_equal(fractions, 0).T,
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
        cmap=CMAP,
        norm=norm,
        interpolation="nearest",
    )
    ax.plot([0, 1], [0, 1], color=DIAGONAL, lw=1.0)
    a, b = _label(pair.a), _label(pair.b)
    corner = {"fontsize": 6, "color": INK_MUTED, "transform": ax.transAxes}
    ax.text(0.97, 0.03, f"{a} dominates", ha="right", va="bottom", **corner)
    ax.text(0.03, 0.97, f"{b} dominates", ha="left", va="top", **corner)
    ticks = np.linspace(0, 1, 6)
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_xlabel(f"solve rate, {a}", fontsize=7)
    ax.set_ylabel(f"solve rate, {b}", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f"{title}\n{stats_text(pair)}", fontsize=6.5, loc="left")
    return image


def _colorbar(fig: Any, image: Any, axes: Any) -> None:
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.03)
    bar.set_label("fraction of levels (log)", fontsize=7)
    bar.ax.tick_params(labelsize=6)


def plot_pairs(report: Report, out: Path) -> list[Path]:
    """The method_a vs method_b heatmap: one grid PDF with a panel per
    robot, plus a PNG and PDF per robot, on one colour scale."""
    a, b = report.pair
    robots = {r: rep.pair for r, rep in report.robots.items() if rep.pair}
    norm = shared_norm(list(robots.values()))
    written = []
    fig, axes = plt.subplots(
        1,
        len(robots),
        figsize=(PANEL * len(robots) + 1.0, PANEL + 0.9),
        squeeze=False,
        layout="constrained",
    )
    for ax, (robot, pair) in zip(axes[0], robots.items(), strict=True):
        image = draw_pair(ax, pair, norm, ROBOT_LABEL[robot])
    _colorbar(fig, image, axes)
    written.append(out / f"domination_{a}_vs_{b}.pdf")
    fig.savefig(written[-1])
    plt.close(fig)
    for robot, pair in robots.items():
        fig, ax = plt.subplots(
            figsize=(PANEL + 1.0, PANEL + 0.9), layout="constrained"
        )
        _colorbar(fig, draw_pair(ax, pair, norm, ROBOT_LABEL[robot]), ax)
        for suffix in ("pdf", "png"):
            written.append(out / f"domination_{a}_vs_{b}_{robot}.{suffix}")
            fig.savefig(written[-1], dpi=200)
        plt.close(fig)
    return written


def draw_matrix(ax: Any, robot: str, report: RobotReport) -> Any:
    methods = report.matrix.methods
    values = np.asarray(report.matrix.row_better)
    image = ax.imshow(values, cmap=CMAP, norm=Normalize(0, 1))
    for i, j in np.ndindex(values.shape):
        if i != j:
            ink = "white" if values[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{values[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=5.5,
                color=ink,
            )
    labels = [_label(m) for m in methods]
    ax.set_xticks(range(len(methods)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)), labels)
    ax.tick_params(labelsize=6, length=0)
    ax.set_title(ROBOT_LABEL[robot], fontsize=8)
    return image


def plot_matrices(report: Report, out: Path) -> Path:
    """Per robot: the fraction of levels where the row method's solve rate
    beats the column method's."""
    robots = report.robots
    fig, axes = plt.subplots(
        1,
        len(robots),
        figsize=(PANEL * len(robots) + 1.0, PANEL + 0.6),
        squeeze=False,
        layout="constrained",
    )
    for ax, (robot, rep) in zip(axes[0], robots.items(), strict=True):
        image = draw_matrix(ax, robot, rep)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, pad=0.03)
    bar.set_label("levels where row beats column", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    path = out / "domination_matrix.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path
