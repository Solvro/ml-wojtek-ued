import dataclasses
import math
import re
import warnings
from pathlib import Path

import numpy as np

from config.schema import DominationConfig, Robot
from evaluation.stats import bootstrap_ci
from paper.labels import method_order
from paper.runs import run_prefixes
from robots.scale import min_travel

# seed means of k/n fractions can differ by float rounding alone
TIE_TOL = 1e-9


@dataclasses.dataclass
class Dump:
    path: Path
    fell: np.ndarray
    travel: np.ndarray
    levels: np.ndarray


@dataclasses.dataclass
class MethodRates:
    seeds: list[int]
    episodes_per_level: int
    rates: np.ndarray


@dataclasses.dataclass
class PairStats:
    a: str
    b: str
    a_better: float
    b_better: float
    ties: float
    mean_diff: float
    mean_diff_ci: tuple[float, float]
    mean_abs_diff: float
    noise_abs_diff: float
    counts: list[list[int]]


@dataclasses.dataclass
class Matrix:
    methods: list[str]
    row_better: list[list[float]]
    ties: list[list[float]]


@dataclasses.dataclass
class MethodSummary:
    seeds: list[int]
    episodes_per_level: int
    mean_solve_rate: float


@dataclasses.dataclass
class RobotReport:
    prefix: str
    num_levels: int
    methods: dict[str, MethodSummary]
    missing_seeds: dict[str, list[int]]
    missing_methods: list[str]
    pair: PairStats | None
    matrix: Matrix


@dataclasses.dataclass
class Report:
    level_set: str
    min_travel: float
    bins: int
    pair: tuple[str, str]
    robots: dict[str, RobotReport]


def find_dumps(eval_dir: Path, prefix: str) -> dict[str, dict[int, Path]]:
    """method -> seed -> `<prefix>_<method>_seed<k>_final_eval.npz`."""
    pattern = re.compile(rf"{re.escape(prefix)}_(.+)_seed(\d+)_final_eval")
    dumps: dict[str, dict[int, Path]] = {}
    for path in sorted(eval_dir.glob(f"{prefix}_*_final_eval.npz")):
        match = pattern.fullmatch(path.stem)
        if match is not None:
            dumps.setdefault(match[1], {})[int(match[2])] = path
    if not dumps:
        raise FileNotFoundError(
            f"no {prefix}_<method>_seed<k>_final_eval.npz in {eval_dir}; "
            "check eval_dir and the prefixes.<robot>=<prefix> overrides"
        )
    return dumps


def levels_key(level_set: str) -> str:
    # every random/d<d> set shrinks the same base genomes, stored once
    if level_set.startswith("random_d"):
        return "random_levels"
    return f"{level_set}_levels"


def load_dump(path: Path, level_set: str) -> Dump:
    with np.load(path) as data:
        raw = dict(data)
    keys = (f"{level_set}_fell", f"{level_set}_travel", levels_key(level_set))
    missing = [k for k in keys if k not in raw]
    if missing:
        sets = sorted(
            k.removesuffix("_fell") for k in raw if k.endswith("_fell")
        )
        raise KeyError(
            f"{path} lacks {missing} for level_set={level_set}; "
            f"per-level sets in it: {sets}. param_uniform needs a "
            "level_space=param eval_brax.py run"
        )
    return Dump(path, raw[keys[0]], raw[keys[1]], raw[keys[2]])


def check_same_levels(dumps: list[Dump]) -> None:
    first = dumps[0]
    for other in dumps[1:]:
        if not np.array_equal(first.levels, other.levels):
            raise ValueError(
                f"{first.path.name} and {other.path.name} were evaluated on "
                "different levels, so they cannot be compared level by "
                "level; rerun eval_brax.py with the same seed, num_levels "
                "and level_space"
            )


def solve_rate(
    fell: np.ndarray, travel: np.ndarray, threshold: float
) -> np.ndarray:
    """Per level (axis 1 of pass x level x episode): fraction of episodes
    with no fall and travel >= threshold, over both passes."""
    return ((fell == 0) & (travel >= threshold)).mean(axis=(0, 2))


def method_rates(dumps: dict[int, Dump], threshold: float) -> MethodRates:
    """Solve rate per level, averaged over the method's seeds."""
    seeds = sorted(dumps)
    per_seed = [
        solve_rate(dumps[s].fell, dumps[s].travel, threshold) for s in seeds
    ]
    episodes = sum(d.fell.shape[0] * d.fell.shape[2] for d in dumps.values())
    return MethodRates(seeds, episodes, np.mean(per_seed, axis=0))


def histogram(a: np.ndarray, b: np.ndarray, bins: int) -> np.ndarray:
    """Level counts, A's bin on axis 0; bin i = [i/bins, (i+1)/bins), the
    last bin closed."""

    def index(rate: np.ndarray) -> np.ndarray:
        # 12/40 * 10 lands just below 3.0 in floats; nudge before flooring
        return np.clip(np.floor(rate * bins + TIE_TOL), 0, bins - 1).astype(int)

    flat = np.bincount(index(a) * bins + index(b), minlength=bins * bins)
    return flat.reshape(bins, bins)


def binomial_pmf(n: int, p: np.ndarray) -> np.ndarray:
    """level x k: P(k of n successes) at each level's rate p."""
    k = np.arange(n + 1)
    choose = np.array([math.comb(n, i) for i in k], dtype=float)
    return choose * p[:, None] ** k * (1 - p[:, None]) ** (n - k)


def noise_abs_diff(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int) -> float:
    """Mean |A - B| two identical policies (per-level rate (a + b) / 2)
    would show from episode sampling alone, with n_a and n_b episodes per
    level: exact binomial expectation, averaged over levels."""
    p = 0.5 * (a + b)
    pmf_a, pmf_b = binomial_pmf(n_a, p), binomial_pmf(n_b, p)
    gap = np.abs(
        np.arange(n_a + 1)[:, None] / n_a - np.arange(n_b + 1)[None] / n_b
    )
    return float(np.einsum("li,lj,ij->l", pmf_a, pmf_b, gap).mean())


def compare_pair(
    names: tuple[str, str], a: MethodRates, b: MethodRates, bins: int
) -> PairStats:
    diff = a.rates - b.rates
    rng = np.random.default_rng(0)
    return PairStats(
        a=names[0],
        b=names[1],
        a_better=float((diff > TIE_TOL).mean()),
        b_better=float((diff < -TIE_TOL).mean()),
        ties=float((np.abs(diff) <= TIE_TOL).mean()),
        mean_diff=float(diff.mean()),
        mean_diff_ci=bootstrap_ci(
            lambda i: float(diff[i].mean()), len(diff), rng
        ),
        mean_abs_diff=float(np.abs(diff).mean()),
        noise_abs_diff=noise_abs_diff(
            a.rates, b.rates, a.episodes_per_level, b.episodes_per_level
        ),
        counts=histogram(a.rates, b.rates, bins).tolist(),
    )


def domination_matrix(rates: dict[str, np.ndarray]) -> Matrix:
    """row_better[i][j]: fraction of levels where method i's rate beats
    method j's; ties[i][j]: equal rates."""
    methods = list(rates)
    diff = np.stack([rates[m] for m in methods])
    diff = diff[:, None, :] - diff[None, :, :]
    return Matrix(
        methods=methods,
        row_better=(diff > TIE_TOL).mean(axis=2).tolist(),
        ties=(np.abs(diff) <= TIE_TOL).mean(axis=2).tolist(),
    )


def missing_seeds(found: dict[str, dict[int, Path]]) -> dict[str, list[int]]:
    """Per method, the seeds some other method of the robot has and it
    lacks."""
    union = set().union(*found.values())
    gaps = {m: sorted(union - set(seeds)) for m, seeds in found.items()}
    return {m: gap for m, gap in gaps.items() if gap}


def robot_report(
    robot: str,
    prefix: str,
    found: dict[str, dict[int, Path]],
    all_methods: list[str],
    cfg: DominationConfig,
) -> RobotReport:
    dumps = {
        m: {s: load_dump(p, cfg.level_set) for s, p in seeds.items()}
        for m, seeds in found.items()
    }
    check_same_levels([d for seeds in dumps.values() for d in seeds.values()])
    threshold = min_travel(Robot(robot), cfg.min_travel)
    rates = {
        m: method_rates(dumps[m], threshold) for m in all_methods if m in dumps
    }
    pair = None
    if cfg.method_a in rates and cfg.method_b in rates:
        pair = compare_pair(
            (cfg.method_a, cfg.method_b),
            rates[cfg.method_a],
            rates[cfg.method_b],
            cfg.bins,
        )
    return RobotReport(
        prefix=prefix,
        num_levels=len(next(iter(rates.values())).rates),
        methods={
            m: MethodSummary(
                r.seeds, r.episodes_per_level, float(r.rates.mean())
            )
            for m, r in rates.items()
        },
        missing_seeds=missing_seeds(found),
        missing_methods=[m for m in all_methods if m not in rates],
        pair=pair,
        matrix=domination_matrix({m: r.rates for m, r in rates.items()}),
    )


def warn_gaps(robot: str, report: RobotReport, cfg: DominationConfig) -> None:
    for method, seeds in report.missing_seeds.items():
        warnings.warn(f"{robot}: {method} lacks seeds {seeds}", stacklevel=2)
    if report.missing_methods:
        warnings.warn(
            f"{robot}: no {report.missing_methods} dumps (other robots have "
            "them)",
            stacklevel=2,
        )
    if report.pair is None:
        warnings.warn(
            f"{robot}: no {cfg.method_a} vs {cfg.method_b} heatmap, one of "
            "them has no dumps",
            stacklevel=2,
        )


def compare(cfg: DominationConfig) -> Report:
    if cfg.method_a == cfg.method_b:
        raise ValueError(
            f"method_a and method_b are both {cfg.method_a}; pick two methods"
        )
    eval_dir = Path(cfg.eval_dir)
    prefixes = run_prefixes(cfg.prefixes)
    found = {r: find_dumps(eval_dir, p) for r, p in prefixes.items()}
    all_methods = method_order({m for ms in found.values() for m in ms})
    robots = {}
    for robot, prefix in prefixes.items():
        robots[robot] = robot_report(
            robot, prefix, found[robot], all_methods, cfg
        )
        warn_gaps(robot, robots[robot], cfg)
    if all(r.pair is None for r in robots.values()):
        raise ValueError(
            f"no robot has both {cfg.method_a} and {cfg.method_b} dumps; "
            f"methods found: {all_methods}. Set method_a / method_b"
        )
    return Report(
        level_set=cfg.level_set,
        min_travel=cfg.min_travel,
        bins=cfg.bins,
        pair=(cfg.method_a, cfg.method_b),
        robots=robots,
    )
