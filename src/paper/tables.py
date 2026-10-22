import csv
import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from config.schema import PaperTablesConfig
from evaluation.final_eval import AXES
from hfield.level_space import PARAMS
from paper.labels import ROBOT_LABEL, method_label, method_order
from paper.latex import by_method, column_cells, tabular
from paper.runs import (
    SOLVABLE,
    Run,
    add_solvable,
    add_train_summaries,
    dump_wandb,
    load_runs,
    run_prefixes,
)

MAIN_COLUMNS = (
    (r"$d{=}0.25$", "random/d0.25/success_rate"),
    ("OOD suite", "suite/success_rate"),
)
# (header, metric key, scale, digits)
DETAIL_COLUMNS = (
    (r"$d{=}0.25$", "random/d0.25/success_rate", 100.0, 1),
    (r"CVaR$_{25}$", "random/d0.25/cvar0.25_success", 100.0, 1),
    (r"$d{=}0.5$", "random/d0.5/success_rate", 100.0, 1),
    ("OOD suite", "suite/success_rate", 100.0, 1),
    ("Flat", "flat/success_rate", 100.0, 1),
)
# (header, metric key, scale, digits)
UNIFORM_COLUMNS = (
    ("Uniform", "param/uniform/success_rate", 100.0, 1),
    (r"CVaR$_{25}$", "param/uniform/cvar0.25_success", 100.0, 1),
    ("Solvable", SOLVABLE, 100.0, 1),
)
# (header, summary key, scale, digits)
COMPUTE_COLUMNS = (
    ("GPU-h", "final/elapsed_h", 1.0, 1),
    (r"Sim steps ($10^6$)", "final/sim_steps", 1e-6, 0),
    (r"Student steps ($10^6$)", "final/student_train_steps", 1e-6, 0),
)

RunsByRobot = dict[str, list[Run]]


def main_table(runs: RunsByRobot) -> str:
    """Success (%) per method, one column group per robot."""
    robots = list(runs)
    methods = method_order({r.method for rs in runs.values() for r in rs})
    columns = [[method_label(m) for m in methods]]
    for robot in robots:
        groups = by_method(runs[robot], methods)
        for _, key in MAIN_COLUMNS:
            columns.append(column_cells(groups, key, 100.0, 1, bold=True))
    n = len(MAIN_COLUMNS)
    groups_head = " & ".join(
        rf"\multicolumn{{{n}}}{{c}}{{{ROBOT_LABEL[robot]}}}" for robot in robots
    )
    rules = "".join(
        rf"\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}"
        for i in range(len(robots))
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
    deg = PARAMS[index].name.endswith("_deg")
    return ("deg", 1.0) if deg else ("cm", 100.0)


def per_robot_table(
    runs: RunsByRobot,
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
        cells = [
            column_cells(groups, key, s, d, bold) for _, key, s, d in columns
        ]
        for j, method in enumerate(methods):
            name = ROBOT_LABEL[robot] if j == 0 else ""
            row = [name, method_label(method), *(c[j] for c in cells)]
            body.append(" & ".join(row) + r" \\")
    return head, body


def envelopes_table(runs: RunsByRobot) -> str:
    """The `axis/<feature>/envelope` values per robot x method."""
    columns = []
    for name, index, _ in AXES:
        unit, scale = axis_unit(index)
        title = name.replace("_", " ")
        columns.append((f"{title} ({unit})", f"axis/{name}/envelope", scale, 1))
    head, body = per_robot_table(runs, columns, bold=True)
    spec = "ll" + "c" * len(columns)
    return tabular(spec, head, body)


def detail_table(runs: RunsByRobot) -> str:
    """Success (%) on every level set per robot x method."""
    head, body = per_robot_table(runs, DETAIL_COLUMNS, bold=True)
    return tabular("ll" + "c" * len(DETAIL_COLUMNS), head, body)


def uniform_table(runs: RunsByRobot) -> str:
    """Success (%) on the `param/uniform` levels per robot x method."""
    head, body = per_robot_table(runs, UNIFORM_COLUMNS, bold=True)
    return tabular("ll" + "c" * len(UNIFORM_COLUMNS), head, body)


def compute_table(runs: RunsByRobot) -> str:
    head, body = per_robot_table(runs, COMPUTE_COLUMNS, bold=False)
    return tabular("llccc", head, body)


def write_csv(runs: RunsByRobot, path: Path) -> None:
    """One row per run, every value the tables use."""
    keys = [key for _, key, _, _ in DETAIL_COLUMNS]
    keys += [key for _, key, _, _ in UNIFORM_COLUMNS]
    keys += [f"axis/{name}/envelope" for name, _, _ in AXES]
    keys += [key for _, key, _, _ in COMPUTE_COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot", "method", "seed", *keys])
        for robot_runs in runs.values():
            for r in robot_runs:
                values = [r.metrics.get(key, "") for key in keys]
                writer.writerow([r.robot, r.method, r.seed, *values])


def tables(cfg: PaperTablesConfig) -> None:
    prefixes = run_prefixes(cfg.prefixes)
    if cfg.dump_wandb:
        dump_wandb(prefixes, Path(cfg.dump_wandb))
        return
    if cfg.eval_dir is None:
        raise ValueError(
            "set eval_dir=<dir of eval_brax.py outputs>, or dump_wandb"
        )
    summaries: dict[str, Any] = {}
    if cfg.train_summaries:
        summaries = json.loads(Path(cfg.train_summaries).read_text())
    runs = {}
    for robot, prefix in prefixes.items():
        runs[robot] = load_runs(Path(cfg.eval_dir), robot, prefix)
        add_solvable(runs[robot])
        add_train_summaries(runs[robot], prefix, summaries)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "main.tex").write_text(main_table(runs))
    (out / "detail.tex").write_text(detail_table(runs))
    (out / "uniform.tex").write_text(uniform_table(runs))
    (out / "envelopes.tex").write_text(envelopes_table(runs))
    if summaries:
        (out / "compute.tex").write_text(compute_table(runs))
    else:
        warnings.warn("no train_summaries: compute.tex not written")
    write_csv(runs, out / "results.csv")
    print(f"wrote tables for {sum(map(len, runs.values()))} runs to {out}")
