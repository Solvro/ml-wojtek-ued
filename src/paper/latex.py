import numpy as np

from paper.runs import Run

Stat = tuple[float, float] | None


def seed_stats(runs: list[Run], key: str) -> Stat:
    """(mean, std) of `key` over the seeds that logged it."""
    values = [run.metrics[key] for run in runs if key in run.metrics]
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std


def cell(stat: Stat, scale: float, digits: int, best: float | None) -> str:
    if stat is None:
        return "--"
    mean = round(stat[0] * scale, digits)
    text = f"{mean:.{digits}f}"
    if best is not None and mean == best:
        text = rf"\textbf{{{text}}}"
    return rf"{text}\,{{\scriptsize$\pm${stat[1] * scale:.{digits}f}}}"


def column_cells(
    groups: list[list[Run]], key: str, scale: float, digits: int, bold: bool
) -> list[str]:
    """One cell per group of runs (a method), best rounded mean in bold."""
    stats = [seed_stats(runs, key) if runs else None for runs in groups]
    means = [round(s[0] * scale, digits) for s in stats if s is not None]
    best = max(means) if bold and len(means) > 1 else None
    return [cell(s, scale, digits, best) for s in stats]


def by_method(runs: list[Run], methods: list[str]) -> list[list[Run]]:
    return [[r for r in runs if r.method == m] for m in methods]


def tabular(spec: str, head: list[str], body: list[str]) -> str:
    """A bare booktabs tabular: the paper owns float, caption and label."""
    return "\n".join(
        [
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            *head,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
