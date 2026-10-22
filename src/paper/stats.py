import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from rliable import library, metrics

from config.schema import PaperStatsConfig
from paper.labels import method_label, method_order
from paper.latex import tabular
from paper.runs import Run, load_runs, run_prefixes
from paper.stats_plots import plot_iqm, plot_poi

METRICS = (
    (r"$d{=}0.25$", "random/d0.25/success_rate"),
    ("OOD suite", "suite/success_rate"),
)

# robot -> method -> seed -> score
Scores = dict[str, dict[str, dict[int, float]]]
Estimate = dict[str, Any]


def collect(runs: dict[str, list[Run]], key: str) -> Scores:
    """The finite `key` values per robot, method and seed."""
    scores: Scores = {}
    for robot, robot_runs in runs.items():
        scores[robot] = {}
        for run in robot_runs:
            value = run.metrics.get(key)
            if value is None or not np.isfinite(value):
                warnings.warn(f"{robot} {run.train_name}: no {key}, skipped")
                continue
            scores[robot].setdefault(run.method, {})[run.seed] = float(value)
    return scores


def score_matrix(scores: Scores, method: str, robots: list[str]) -> np.ndarray:
    """(runs, robots) for rliable; robots with more seeds keep the lowest
    `runs` of them (the bootstrap resamples runs per robot, so seed numbers
    need not match across robots)."""
    counts = [len(scores[robot][method]) for robot in robots]
    runs = min(counts)
    if runs < max(counts):
        warnings.warn(
            f"{method}: {sum(counts) - runs * len(robots)} runs dropped to "
            f"keep {runs} seeds on each of {robots}"
        )
    columns = [
        [scores[robot][method][seed] for seed in sorted(scores[robot][method])]
        for robot in robots
    ]
    return np.array([column[:runs] for column in columns]).T


def iqm_and_mean(scores: np.ndarray) -> np.ndarray:
    return np.array(
        [metrics.aggregate_iqm(scores), metrics.aggregate_mean(scores)]
    )


def estimate(scores: np.ndarray, reps: int) -> Estimate:
    """IQM and mean with 95 % stratified bootstrap CIs (strata = columns)."""
    points, cis = library.get_interval_estimates(
        {"x": scores}, iqm_and_mean, reps=reps
    )
    (iqm, mean), ci = points["x"], cis["x"]
    return {
        "runs": int(scores.size),
        "iqm": float(iqm),
        "iqm_ci": [float(ci[0, 0]), float(ci[1, 0])],
        "mean": float(mean),
        "mean_ci": [float(ci[0, 1]), float(ci[1, 1])],
    }


def per_robot(
    scores: Scores, methods: list[str], reps: int
) -> dict[str, dict[str, Estimate]]:
    result: dict[str, dict[str, Estimate]] = {}
    for robot, by_method in scores.items():
        result[robot] = {}
        for method in methods:
            if method in by_method:
                matrix = score_matrix(scores, method, [robot])
                result[robot][method] = {
                    "seeds": sorted(by_method[method]),
                    **estimate(matrix, reps),
                }
    return result


def normalisation(scores: Scores, key: str) -> dict[str, dict[str, Any]]:
    """Per robot, min and max of the method means (seed mean)."""
    result = {}
    for robot, by_method in scores.items():
        if not by_method:
            raise ValueError(
                f"{robot}: no run has {key}; rerun eval_brax.py on its "
                f"checkpoints or leave it out with prefixes.{robot}=null"
            )
        means = {
            m: float(np.mean(list(s.values()))) for m, s in by_method.items()
        }
        lo = min(means, key=means.__getitem__)
        hi = max(means, key=means.__getitem__)
        if means[hi] == means[lo]:
            raise ValueError(
                f"{robot}: every method ({sorted(means)}) has the same mean "
                f"{key} = {means[lo]}, so min-max normalisation is undefined; "
                f"leave the robot out with prefixes.{robot}=null"
            )
        result[robot] = {
            "min": means[lo],
            "max": means[hi],
            "min_method": lo,
            "max_method": hi,
        }
    return result


def aggregate(
    scores: Scores,
    norm: dict[str, dict[str, Any]],
    methods: list[str],
    reps: int,
) -> dict[str, Estimate]:
    """Estimates over all robot x seed runs of the min-max normalised scores,
    for the methods with runs on every robot."""
    robots = list(scores)
    lo = np.array([norm[robot]["min"] for robot in robots])
    hi = np.array([norm[robot]["max"] for robot in robots])
    result = {}
    for method in methods:
        if all(method in scores[robot] for robot in robots):
            matrix = score_matrix(scores, method, robots)
            result[method] = estimate((matrix - lo) / (hi - lo), reps)
    return result


def probability_of_improvement(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """rliable's `metrics.probability_of_improvement` on (runs, robots)
    matrices, vectorised: equal run counts per robot make the mean over
    all (x run, y run, robot) triples the robot mean of U / (n m)."""
    wins = x[:, None, :] > y[None, :, :]
    ties = x[:, None, :] == y[None, :, :]
    return np.array([np.mean(wins + 0.5 * ties)])


def improvement(scores: Scores, x: str, y: str, reps: int) -> Estimate | None:
    """P(X > Y) over the robots both methods have runs on: per robot the
    Mann-Whitney U / (n m), averaged over robots; None without such a robot."""
    robots = [r for r, by in scores.items() if x in by and y in by]
    if not robots:
        return None
    pair = [score_matrix(scores, x, robots), score_matrix(scores, y, robots)]
    points, cis = library.get_interval_estimates(
        {"p": pair}, probability_of_improvement, reps=reps
    )
    lo, hi = np.ravel(cis["p"])
    return {
        "p": float(points["p"][0]),
        "ci": [float(lo), float(hi)],
        "robots": robots,
    }


def all_improvements(
    scores: Scores, methods: list[str], reps: int
) -> dict[str, dict[str, Estimate]]:
    """P(X > Y) for every ordered pair; P(Y > X) = 1 - P(X > Y), as
    Mann-Whitney U counts ties as 1/2, with the CI mirrored."""
    result: dict[str, dict[str, Estimate]] = {m: {} for m in methods}
    for x, y in itertools.combinations(methods, 2):
        forward = improvement(scores, x, y, reps)
        if forward is None:
            continue
        lo, hi = forward["ci"]
        result[x][y] = forward
        result[y][x] = {
            **forward,
            "p": 1 - forward["p"],
            "ci": [1 - hi, 1 - lo],
        }
    return result


def metric_stats(
    scores: Scores,
    metric: tuple[str, str],
    methods: list[str],
    cfg: PaperStatsConfig,
) -> dict[str, Any]:
    name, key = metric
    norm = normalisation(scores, key)
    robots = list(scores)
    missing = {
        m: [r for r in robots if m not in scores[r]]
        for m in methods
        if not all(m in scores[r] for r in robots)
    }
    for method, absent in missing.items():
        warnings.warn(
            f"{key}: {method} has no runs on {absent}; left out of the "
            "aggregate, its P(X > Y) uses the robots it has"
        )
    poi = all_improvements(scores, methods, cfg.poi_reps)
    named = [(x, y) for x, y in cfg.pairs if x in poi and y in poi[x]]
    return {
        "name": name,
        "per_robot": per_robot(scores, methods, cfg.reps),
        "normalisation": norm,
        "missing": missing,
        "aggregate": aggregate(scores, norm, methods, cfg.reps),
        "poi": poi,
        "named_pairs": [{"x": x, "y": y, **poi[x][y]} for x, y in named],
        "skipped_pairs": [p for p in cfg.pairs if tuple(p) not in named],
    }


def ci_cell(e: Estimate | None) -> str:
    if e is None:
        return "--"
    lo, hi = e["iqm_ci"]
    return rf"{e['iqm']:.2f}\,{{\scriptsize[{lo:.2f}, {hi:.2f}]}}"


def stats_table(report: dict[str, Any]) -> str:
    """Methods x aggregate IQM [95 % CI] per metric."""
    names = " & ".join(f"IQM {name}" for name, _ in METRICS)
    head = [f"Method & {names} " + r"\\"]
    body = []
    for method in report["methods"]:
        cells = [
            ci_cell(report["metrics"][key]["aggregate"].get(method))
            for _, key in METRICS
        ]
        body.append(" & ".join([method_label(method), *cells]) + r" \\")
    return tabular("l" + "c" * len(METRICS), head, body)


def stats(cfg: PaperStatsConfig) -> None:
    prefixes = run_prefixes(cfg.prefixes)
    eval_dir = Path(cfg.eval_dir)
    runs = {r: load_runs(eval_dir, r, p) for r, p in prefixes.items()}
    methods = method_order({run.method for rs in runs.values() for run in rs})
    # rliable's bootstraps draw from numpy's global RNG, not random_state
    np.random.seed(cfg.seed)
    report = {
        "robots": list(runs),
        "methods": methods,
        "labels": {m: method_label(m) for m in methods},
        "reps": cfg.reps,
        "poi_reps": cfg.poi_reps,
        "seed": cfg.seed,
        "metrics": {
            key: metric_stats(collect(runs, key), (name, key), methods, cfg)
            for name, key in METRICS
        },
    }
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(report, indent=2))
    (out / "stats.tex").write_text(stats_table(report))
    written = [out / "stats.json", out / "stats.tex", plot_iqm(report, out)]
    written += plot_poi(report, out)
    for key, values in report["metrics"].items():
        if values["skipped_pairs"]:
            print(f"{key}: no runs of both, skipped {values['skipped_pairs']}")
    print("\n".join(f"wrote {p}" for p in written))
