import dataclasses
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from config.schema import Robot
from robots.scale import ROBOT_SCALE

ENTITY = "dominikgalus9-wroc-aw-university-of-science-and-technology"
PROJECT = "wojtyla"
MIN_TRAVEL_M = 1.0  # eval_brax.py min_travel, at Wojtek's size
SOLVABLE = "param/uniform_solvable/success_rate"


@dataclasses.dataclass
class Run:
    robot: str
    method: str
    seed: int
    metrics: dict[str, float]
    raw: dict[str, np.ndarray]

    @property
    def train_name(self) -> str:
        return f"{self.method}_seed{self.seed}"


def run_prefixes(prefixes: dict[Robot, str | None]) -> dict[str, str]:
    """The robots with a run prefix set, in config order."""
    chosen = {
        str(robot): prefix for robot, prefix in prefixes.items() if prefix
    }
    if not chosen:
        raise ValueError(
            "no run prefix set; pass prefixes.<robot>=<prefix>, e.g. "
            "prefixes.wojtek=v10_wojtek prefixes.go1=v10_go1"
        )
    return chosen


def load_runs(eval_dir: Path, robot: str, prefix: str) -> list[Run]:
    """Every `<prefix>_<method>_seed<k>_final_eval.{json,npz}` in eval_dir."""
    pattern = re.compile(rf"{re.escape(prefix)}_(.+)_seed(\d+)_final_eval")
    runs = []
    for path in sorted(eval_dir.glob(f"{prefix}_*_final_eval.json")):
        match = pattern.fullmatch(path.stem)
        if match is None:
            continue
        npz = path.with_suffix(".npz")
        if not npz.exists():
            warnings.warn(f"{npz} missing: per-level tables skip this run")
        raw = dict(np.load(npz)) if npz.exists() else {}
        metrics = json.loads(path.read_text())
        runs.append(Run(robot, match[1], int(match[2]), metrics, raw))
    if not runs:
        raise FileNotFoundError(
            f"no {prefix}_<method>_seed<k>_final_eval.json in {eval_dir}; "
            "check eval_dir and the prefixes.<robot>=<prefix> overrides"
        )
    return runs


def level_success(run: Run, pass_index: int, min_travel: float) -> np.ndarray:
    fell = run.raw["param_uniform_fell"][pass_index]
    travel = run.raw["param_uniform_travel"][pass_index]
    return ((fell == 0) & (travel >= min_travel)).mean(axis=1)


def check_same_levels(runs: list[Run]) -> None:
    levels = [
        r.raw["param_uniform_levels"]
        for r in runs
        if "param_uniform_levels" in r.raw
    ]
    if any(not np.array_equal(levels[0], other) for other in levels[1:]):
        raise ValueError(
            f"{runs[0].robot}: runs were evaluated on different param/uniform "
            "levels; rerun eval_brax.py with the same seed and num_levels"
        )


def add_solvable(runs: list[Run]) -> None:
    """Success on the `param/uniform` levels some run of the robot passes:
    selected on one pass, scored on the other, both ways."""
    robot = runs[0].robot
    needed = {"param_uniform_fell", "param_uniform_travel"}
    old = [r for r in runs if not needed <= r.raw.keys()]
    if old:
        warnings.warn(
            f"{robot}: {len(old)}/{len(runs)} runs lack param_uniform_travel "
            "(pre-v10 eval); solvable-subset success left as --"
        )
        return
    check_same_levels(runs)
    min_travel = MIN_TRAVEL_M * ROBOT_SCALE[robot]
    success = np.stack(
        [[level_success(r, i, min_travel) for i in (0, 1)] for r in runs]
    )
    solvable = (success > 0).any(axis=0)
    if not solvable.any(axis=1).all():
        warnings.warn(
            f"{robot}: no run passes any uniform level; solvable left --"
        )
        return
    for run, (a, b) in zip(runs, success, strict=True):
        run.metrics[SOLVABLE] = float(
            0.5 * (b[solvable[0]].mean() + a[solvable[1]].mean())
        )


def add_train_summaries(
    runs: list[Run], prefix: str, summaries: dict[str, Any]
) -> None:
    for run in runs:
        summary = summaries.get(f"{prefix}_{run.train_name}", {})
        finals = {k: v for k, v in summary.items() if k.startswith("final/")}
        run.metrics.update(finals)


def dump_wandb(prefixes: dict[str, str], path: Path) -> None:
    """The training-run summaries of `prefixes`, as `train_summaries` JSON."""
    api = wandb.Api()
    summaries: dict[str, Any] = {}
    for prefix in prefixes.values():
        name = rf"^{re.escape(prefix)}_.+_seed[0-9]+$"
        for run in api.runs(
            f"{ENTITY}/{PROJECT}", filters={"display_name": {"$regex": name}}
        ):
            summary = dict(run.summary._json_dict)
            # a crashed attempt can share the name; keep the one that finished
            if run.name not in summaries or "final/elapsed_h" in summary:
                summaries[run.name] = summary
    path.write_text(json.dumps(summaries, indent=2))
    print(f"wrote {len(summaries)} run summaries to {path}")
