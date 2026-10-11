"""Paper tables from `eval_brax.py` final evals and training summaries.

Reads `<prefix>_<method>_seed<k>_final_eval.{json,npz}` from `--eval_dir`
for every `--prefix robot=prefix` and writes bare booktabs `tabular`s
(the paper owns float, caption and label) plus a per-run CSV to `--out`:

- `main.tex`: success (%) per method, column groups per robot: uniform,
  CVaR25, solvable subset, OOD suite, flat; mean +- std over seeds, best
  per column in bold.
- `envelopes.tex`: the 7 `axis/<feature>/envelope` values per robot x method.
- `compute.tex`: GPU-hours and simulated / student-trained env steps, from
  `--train_summaries` (run name -> wandb summary, see `--dump_wandb`).
- `results.csv`: one row per run, every value the tables use.

Solvable subset: `param/uniform` levels that at least one run of the robot
(any method) passes on at least one episode of pass A; each run is scored
on pass B on that subset and vice versa, averaged, so selection noise does
not inflate it. Needs the per-level `param_uniform_travel` (v10 evals).

`--dump_wandb FILE` fetches the training-run summaries for the prefixes from
wandb and writes the JSON `--train_summaries` reads, then exits.
"""

import argparse
import csv
import dataclasses
import json
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from hfield.level_space import PARAMS
from scripts.brax_ued import ROBOT_SCALE
from scripts.eval_brax import AXES

ENTITY = "dominikgalus9-wroc-aw-university-of-science-and-technology"
PROJECT = "wojtyla"
MIN_TRAVEL_M = 1.0  # eval_brax.py --min_travel, at Wojtek's size
METHODS = {
    "dr": "DR",
    "dr_easy": "DR-easy",
    "plr": r"PLR$^{\perp}$",
    "accel": "ACCEL",
    "accel_sfl": "ACCEL-SFL",
    "paired": "PAIRED",
}
SOLVABLE = "param/uniform_solvable/success_rate"
MAIN_COLUMNS = (
    ("Uniform", "param/uniform/success_rate"),
    (r"CVaR$_{25}$", "param/uniform/cvar0.25_success"),
    ("Solvable", SOLVABLE),
    ("OOD suite", "suite/success_rate"),
    ("Flat", "flat/success_rate"),
)
COMPUTE_COLUMNS = (
    ("GPU-h", "final/elapsed_h", 1.0, 1),
    (r"Sim steps ($10^6$)", "final/sim_steps", 1e-6, 0),
    (r"Student steps ($10^6$)", "final/student_train_steps", 1e-6, 0),
)


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


def method_label(method: str) -> str:
    return METHODS.get(method, method.replace("_", r"\_"))


def method_order(methods: set[str]) -> list[str]:
    known = [m for m in METHODS if m in methods]
    return known + sorted(methods - set(METHODS))


def load_runs(eval_dir: Path, robot: str, prefix: str) -> list[Run]:
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
            "check --eval_dir and the --prefix robot=prefix pairs"
        )
    return runs


def level_success(run: Run, pass_index: int, min_travel: float) -> np.ndarray:
    fell = run.raw["param_uniform_fell"][pass_index]
    travel = run.raw["param_uniform_travel"][pass_index]
    return ((fell == 0) & (travel >= min_travel)).mean(axis=1)


def add_solvable(runs: list[Run]) -> None:
    robot = runs[0].robot
    needed = {"param_uniform_fell", "param_uniform_travel"}
    old = [r for r in runs if not needed <= r.raw.keys()]
    if old:
        warnings.warn(
            f"{robot}: {len(old)}/{len(runs)} runs lack param_uniform_travel "
            "(pre-v10 eval); solvable-subset success left as --"
        )
        return
    levels = [
        r.raw["param_uniform_levels"] for r in runs if "param_uniform_levels" in r.raw
    ]
    if any(not np.array_equal(levels[0], other) for other in levels[1:]):
        raise ValueError(
            f"{robot}: runs were evaluated on different param/uniform levels; "
            "rerun eval_brax.py with the same --seed and --num_levels"
        )
    min_travel = MIN_TRAVEL_M * ROBOT_SCALE[robot]
    success = np.stack(
        [[level_success(r, i, min_travel) for i in (0, 1)] for r in runs]
    )
    solvable = (success > 0).any(axis=0)
    if not solvable.any(axis=1).all():
        warnings.warn(f"{robot}: no run passes any uniform level; solvable left --")
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
        run.metrics.update({k: v for k, v in summary.items() if k.startswith("final/")})


def seed_stats(runs: list[Run], key: str) -> tuple[float, float] | None:
    values = [run.metrics[key] for run in runs if key in run.metrics]
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return float(np.mean(values)), std


def cell(
    stat: tuple[float, float] | None, scale: float, digits: int, best: float | None
) -> str:
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


def main_table(runs: dict[str, list[Run]]) -> str:
    robots = list(runs)
    methods = method_order({r.method for rs in runs.values() for r in rs})
    columns = [[method_label(m) for m in methods]]
    for robot in robots:
        groups = by_method(runs[robot], methods)
        for _, key in MAIN_COLUMNS:
            columns.append(column_cells(groups, key, 100.0, 1, bold=True))
    n = len(MAIN_COLUMNS)
    groups_head = " & ".join(
        rf"\multicolumn{{{n}}}{{c}}{{{robot.capitalize()}}}" for robot in robots
    )
    rules = "".join(
        rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(robots))
    )
    names = " & ".join(name for name, _ in MAIN_COLUMNS)
    head = [
        rf" & {groups_head} \\",
        rules,
        "Method & " + " & ".join([names] * len(robots)) + r" \\",
    ]
    body = [" & ".join(row) + r" \\" for row in zip(*columns, strict=True)]
    spec = "l" + " ".join(["c" * n] * len(robots))
    return tabular(spec, head, body)


def axis_unit(index: int) -> tuple[str, float]:
    return ("deg", 1.0) if PARAMS[index].name.endswith("_deg") else ("cm", 100.0)


def per_robot_table(
    runs: dict[str, list[Run]],
    columns: Sequence[tuple[str, str, float, int]],
    bold: bool,
) -> tuple[list[str], list[str]]:
    """Rows robot x method; returns (header row, body rows)."""
    head = ["Robot & Method & " + " & ".join(c[0] for c in columns) + r" \\"]
    body = []
    for i, (robot, robot_runs) in enumerate(runs.items()):
        if i:
            body.append(r"\midrule")
        methods = method_order({r.method for r in robot_runs})
        groups = by_method(robot_runs, methods)
        cells = [column_cells(groups, key, s, d, bold) for _, key, s, d in columns]
        for j, method in enumerate(methods):
            name = robot.capitalize() if j == 0 else ""
            row = [name, method_label(method), *(c[j] for c in cells)]
            body.append(" & ".join(row) + r" \\")
    return head, body


def envelopes_table(runs: dict[str, list[Run]]) -> str:
    columns = []
    for name, index, _ in AXES:
        unit, scale = axis_unit(index)
        title = name.replace("_", " ")
        columns.append((f"{title} ({unit})", f"axis/{name}/envelope", scale, 1))
    head, body = per_robot_table(runs, columns, bold=True)
    spec = "ll" + "c" * len(columns)
    return tabular(spec, head, body)


def compute_table(runs: dict[str, list[Run]]) -> str:
    head, body = per_robot_table(runs, COMPUTE_COLUMNS, bold=False)
    return tabular("llccc", head, body)


def write_csv(runs: dict[str, list[Run]], path: Path) -> None:
    keys = [key for _, key in MAIN_COLUMNS]
    keys += [f"axis/{name}/envelope" for name, _, _ in AXES]
    keys += [key for _, key, _, _ in COMPUTE_COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot", "method", "seed", *keys])
        for robot_runs in runs.values():
            for r in robot_runs:
                values = [r.metrics.get(key, "") for key in keys]
                writer.writerow([r.robot, r.method, r.seed, *values])


def dump_wandb(prefixes: dict[str, str], path: Path) -> None:
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


def parse_prefixes(pairs: list[str]) -> dict[str, str]:
    prefixes = {}
    for pair in pairs:
        robot, _, prefix = pair.partition("=")
        if robot not in ROBOT_SCALE or not prefix:
            raise ValueError(
                f"--prefix {pair!r}: expected robot=prefix with robot in "
                f"{list(ROBOT_SCALE)}, e.g. wojtek=v10_wojtek"
            )
        prefixes[robot] = prefix
    return prefixes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", type=Path)
    p.add_argument("--prefix", nargs="+", action="extend", required=True)
    p.add_argument("--out", type=Path, default=Path("paper/tables"))
    p.add_argument("--train_summaries", type=Path)
    p.add_argument("--dump_wandb", type=Path, help="write summaries JSON, exit")
    args = p.parse_args()
    prefixes = parse_prefixes(args.prefix)
    if args.dump_wandb:
        dump_wandb(prefixes, args.dump_wandb)
        return
    if args.eval_dir is None:
        p.error("--eval_dir is required unless --dump_wandb is given")
    summaries = {}
    if args.train_summaries:
        summaries = json.loads(args.train_summaries.read_text())
    runs = {}
    for robot, prefix in prefixes.items():
        runs[robot] = load_runs(args.eval_dir, robot, prefix)
        add_solvable(runs[robot])
        add_train_summaries(runs[robot], prefix, summaries)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "main.tex").write_text(main_table(runs))
    (args.out / "envelopes.tex").write_text(envelopes_table(runs))
    if summaries:
        (args.out / "compute.tex").write_text(compute_table(runs))
    else:
        warnings.warn("no --train_summaries: compute.tex not written")
    write_csv(runs, args.out / "results.csv")
    print(f"wrote tables for {sum(map(len, runs.values()))} runs to {args.out}")


if __name__ == "__main__":
    main()
