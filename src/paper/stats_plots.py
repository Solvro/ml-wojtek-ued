from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rliable import plot_utils

from paper.figures import COLORS
from paper.labels import method_label

PAIR_SEPARATOR = ";"
POI_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]


def plot_label(method: str) -> str:
    """The paper name as matplotlib text (mathtext, no LaTeX escapes)."""
    return method_label(method).replace(r"\_", "_")


def plot_iqm(report: dict[str, Any], out: Path) -> Path:
    """rliable interval estimates: aggregate IQM and mean per metric."""
    per_metric = list(report["metrics"].values())
    methods = [
        m
        for m in report["methods"]
        if all(m in values["aggregate"] for values in per_metric)
    ]
    names = []
    for values in per_metric:
        names += [f"IQM, {values['name']}", f"Mean, {values['name']}"]
    points, cis, colors = {}, {}, {}
    # rliable draws the first method at the bottom
    for method in reversed(methods):
        label = plot_label(method)
        found = [values["aggregate"][method] for values in per_metric]
        points[label] = np.array(
            [v for e in found for v in (e["iqm"], e["mean"])]
        )
        cis[label] = np.array(
            [v for e in found for v in (e["iqm_ci"], e["mean_ci"])]
        ).T
        colors[label] = COLORS.get(method, "tab:gray")
    fig, _ = plot_utils.plot_interval_estimates(
        points,
        cis,
        metric_names=names,
        algorithms=list(points),
        colors=colors,
        xlabel="Normalised score",
    )
    path = out / "iqm.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_poi(report: dict[str, Any], out: Path) -> list[Path]:
    """rliable probability of improvement of the named pairs, one panel per
    metric; nothing without a named pair."""
    per_metric = [v for v in report["metrics"].values() if v["named_pairs"]]
    if not per_metric:
        return []
    rows = max(len(values["named_pairs"]) for values in per_metric)
    fig, axes = plt.subplots(
        len(per_metric),
        1,
        figsize=(5.0, (0.45 * rows + 1.2) * len(per_metric)),
        squeeze=False,
    )
    for ax, values in zip(axes[:, 0], per_metric, strict=True):
        points, cis = {}, {}
        # rliable draws the first pair at the bottom
        for pair in reversed(values["named_pairs"]):
            key = plot_label(pair["x"]) + PAIR_SEPARATOR + plot_label(pair["y"])
            points[key] = pair["p"]
            cis[key] = pair["ci"]
        plot_utils.plot_probability_of_improvement(
            points,
            cis,
            pair_separator=PAIR_SEPARATOR,
            ax=ax,
            xticks=POI_TICKS,
            xlabel=f"P(X > Y), {values['name']}",
        )
        ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    path = out / "poi.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [path]
