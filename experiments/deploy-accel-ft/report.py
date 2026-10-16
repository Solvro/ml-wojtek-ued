"""report.html of deploy-accel-ft from analyze.py's results and the curves.

    uv run python experiments/deploy-accel-ft/report.py \
        --results results.json --curves cache/curves.json \
        --notes notes.html --out report.html

`--notes`: an HTML fragment (diagnosis, proposals, cost) placed after the
tables; the tables, verdicts and curves come from the data.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

# letter, name, colour slot; ACCEL arms first, DR controls last
ARMS = {
    "accel": ("A", "ACCEL, T6 flags", "s1"),
    "accel_lr1e-4": ("B", "ACCEL, lr 1e-4", "s2"),
    "accel_lr1e-4_1b": ("B→1B", "B continued to 1B", "s2"),
    "accel_hard_lr1e-4": ("D", "ACCEL harder, lr 1e-4", "s4"),
    "accel_sfl_lr1e-4": ("F", "ACCEL SFL, lr 1e-4", "s6"),
    "dr": ("C", "DR control, lr 3e-4", "s3"),
    "dr_lr1e-4": ("E", "DR control, lr 1e-4", "s5"),
}
# B to 1B starts from B's 300M checkpoint
STEP_OFFSET = {"accel_lr1e-4_1b": 300.0}
PRIMARY = [
    ("random/d1.0/cmd_success_rate", "d1.0 cmd success"),
    ("random/d1.0/success_rate", "d1.0 success"),
]
SECONDARY = [
    ("random/d0.5/success_rate", "d0.5 success"),
    ("random/d0.75/success_rate", "d0.75 success"),
    ("param/uniform/success_rate", "param/uniform success"),
    ("param/uniform/cmd_success_rate", "param/uniform cmd"),
]
GUARDS = [
    ("random/d0.25/success_rate", "d0.25 success"),
    ("random/d1.0/fall_rate", "d1.0 fall rate"),
]


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def ci(stats: dict[str, float]) -> str:
    return (
        f'<span class="v">{pct(stats["mean"])}</span>'
        f'<span class="ci">{pct(stats["ci_low"])}–{pct(stats["ci_high"])}</span>'
    )


def diff(stats: dict[str, float]) -> str:
    d, lo, hi = stats["diff"], stats["diff_ci_low"], stats["diff_ci_high"]
    sign = "pos" if lo > 0 else "neg" if hi < 0 else "zero"
    return (
        f'<span class="d {sign}">{100 * d:+.1f}</span>'
        f'<span class="ci">{100 * lo:+.1f}…{100 * hi:+.1f}</span>'
    )


def level_table(section: dict[str, Any], keys: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{label}</th><th>vs base</th>" for _, label in keys)
    rows = []
    base = section["base"]["levels"]
    cells = "".join(f"<td>{ci(base[k]['pooled'])}</td><td></td>" for k, _ in keys)
    rows.append(f'<tr class="base"><th scope="row">Base 1.5B</th>{cells}</tr>')
    for arm, (letter, name, slot) in ARMS.items():
        if arm not in section:
            continue
        stats = section[arm]["levels"]
        cells = "".join(
            f"<td>{ci(stats[k]['pooled'])}</td><td>{diff(stats[k]['pooled'])}</td>"
            for k, _ in keys
        )
        rows.append(
            f'<tr class="arm"><th scope="row"><i class="sw {slot}"></i>{letter} '
            f"{name} <small>3 seeds</small></th>{cells}</tr>"
        )
        for seed, _ in sorted(stats[keys[0][0]]["per_seed"].items()):
            cells = "".join(
                f"<td>{ci(stats[k]['per_seed'][seed])}</td>"
                f"<td>{diff(stats[k]['per_seed'][seed])}</td>"
                for k, _ in keys
            )
            rows.append(
                f'<tr class="seed"><th scope="row">seed {seed}</th>{cells}</tr>'
            )
    return (
        '<div class="tw"><table><thead><tr><th>policy</th>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def json_table(results: dict[str, Any], keys: list[tuple[str, str, str]]) -> str:
    """keys: (section, metric, label) of eval_brax summary metrics."""
    head = "".join(f"<th>{label}</th>" for _, _, label in keys)
    rows = []
    for arm, label in [
        ("base", "Base 1.5B"),
        *[(a, f"{v[0]} {v[1]}") for a, v in ARMS.items()],
    ]:
        cells = []
        for section, metric, _ in keys:
            entry = results[section].get(arm)
            if entry is None:
                cells.append("<td>–</td>")
                continue
            js = entry["json"][metric]
            seeds = " ".join(pct(v) for v in js["per_seed"].values())
            extra = f'<span class="ci">{seeds}</span>' if arm != "base" else ""
            cells.append(f'<td><span class="v">{pct(js["mean"])}</span>{extra}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(cells)}</tr>')
    return (
        f'<div class="tw"><table><thead><tr><th>policy</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def verdict_rows(results: dict[str, Any]) -> str:
    rows = []
    for arm, (letter, name, slot) in ARMS.items():
        entry = results["final"].get(arm)
        if entry is None:
            continue
        for key, v in entry["verdict"].items():
            label = "3-seed pool" if key == "pooled" else key
            checks = [*v["primary"].items(), *v["guards"].items()]
            marks = "".join(
                f'<td class="{"ok" if passed else "no"}">{"pass" if passed else "fail"}</td>'
                for _, passed in checks
            )
            win = (
                '<td class="win yes">beats base</td>'
                if v["win"]
                else '<td class="win">does not</td>'
            )
            rows.append(
                f'<tr class="{"arm" if key == "pooled" else "seed"}"><th scope="row">'
                f'<i class="sw {slot}"></i>{letter} {label}</th>{marks}{win}</tr>'
            )
    first = next(
        iter(
            results["final"][next(a for a in ARMS if a in results["final"])][
                "verdict"
            ].values()
        )
    )
    heads = [*first["primary"], *first["guards"]]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in heads)
    return (
        '<div class="tw"><table class="verdict"><thead><tr><th>arm</th>'
        f"{head}<th>verdict</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def curve_series(curves: dict[str, Any], key: str) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        runs = [
            v["eval"]
            for k, v in curves.items()
            if k.startswith(f"dft2_deploy_{arm}_seed")
        ]
        if not runs:
            continue
        n = min(len(r) for r in runs)
        offset = STEP_OFFSET.get(arm, 0.0)
        x = [offset + runs[0][i]["env_steps"] / 1e6 for i in range(n)]
        ys = np.array([[r[i][key] for i in range(n)] for r in runs])
        out[arm] = {
            "x": x,
            "mean": ys.mean(axis=0).tolist(),
            "lo": ys.min(axis=0).tolist(),
            "hi": ys.max(axis=0).tolist(),
            "seeds": ys.tolist(),
        }
    # iteration 0 of an ACCEL run trains nothing before its eval: the base
    first = [
        v["eval"][0][key]
        for k, v in curves.items()
        if k.startswith("dft2_deploy_accel") and "_1b_" not in k and v["eval"]
    ]
    out["base"] = {"mean": float(np.mean(first)), "lo": min(first), "hi": max(first)}
    return out


def composition_table(curves: dict[str, Any]) -> str:
    """Second half of training: levels trained on (replay iterations = the
    ACCEL buffer's levels), training success, divergence frequency."""
    rows = []
    for arm, (letter, name, slot) in ARMS.items():
        runs = [
            v["train"]
            for k, v in curves.items()
            if k.startswith(f"dft2_deploy_{arm}_seed")
        ]
        if not runs:
            continue
        late = [row for r in runs for row in r[len(r) // 2 :]]
        trained = [row for row in late if row["replay"] == 1]
        dist = np.mean([row["level/genome_dist"] for row in trained])
        success = np.mean([row["train_success"] for row in trained])
        diverged = np.mean([row["diverged_per_step"] > 0 for r in runs for row in r])
        rows.append(
            f'<tr><th scope="row"><i class="sw {slot}"></i>{letter} {name}</th>'
            f'<td><span class="v">{dist:.3f}</span></td>'
            f'<td><span class="v">{pct(success)}</span></td>'
            f'<td><span class="v">{pct(diverged)}</span></td></tr>'
        )
    return (
        '<div class="tw"><table><thead><tr><th>arm</th><th>genome distance from flat</th>'
        "<th>training success %</th><th>iterations with a diverged env %</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--curves", type=Path, required=True)
    p.add_argument("--notes", type=Path, required=True)
    p.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("report_template.html"),
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    results = json.loads(args.results.read_text())
    curves = json.loads(args.curves.read_text())
    charts = {
        "cmd": curve_series(curves, "eval/d1.0/cmd_success_rate"),
        "succ": curve_series(curves, "eval/d1.0/success_rate"),
    }
    page = args.template.read_text()
    page = page.replace("{{VERDICT}}", verdict_rows(results))
    page = page.replace("{{PRIMARY}}", level_table(results["final"], PRIMARY))
    page = page.replace("{{GUARDS}}", level_table(results["final"], GUARDS))
    page = page.replace("{{SECONDARY}}", level_table(results["final"], SECONDARY))
    page = page.replace("{{V10BOX}}", level_table(results["v10box"], PRIMARY))
    page = page.replace(
        "{{SUITE}}",
        json_table(
            results,
            [
                ("final", "suite/success_rate", "suite success, native"),
                ("final", "suite/cmd_success_rate", "suite cmd, native"),
                ("v10box", "suite/success_rate", "suite success, v10box"),
                ("final", "flat/success_rate", "flat success"),
            ],
        ),
    )
    page = page.replace("{{COMPOSITION}}", composition_table(curves))
    legend = "".join(
        f'<span><i class="sw {slot}"></i>{letter} {html.escape(name)}</span>'
        for arm, (letter, name, slot) in ARMS.items()
        if arm in charts["cmd"] and arm != "accel_lr1e-4_1b"
    )
    page = page.replace(
        "{{LEGEND}}", legend + '<span><i class="sw sbase"></i>base</span>'
    )
    meta = {a: {"slot": v[2], "name": f"{v[0]} {v[1]}"} for a, v in ARMS.items()}
    page = page.replace("{{META}}", json.dumps(meta))
    page = page.replace("{{NOTES}}", args.notes.read_text())
    page = page.replace("{{CHARTS}}", json.dumps(charts))
    args.out.write_text(page)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
