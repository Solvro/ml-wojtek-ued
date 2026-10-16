"""Final-eval tables and the pre-registered win verdict of robot-ft.

    PYTHONPATH=src uv run python experiments/robot-ft/analyze.py \
        --runs_dir <local copy of $PDW/runs/robot_ft> --out results.json

Reads every `rft_robot_<arm>_seed<k>_{flat,final,v10box}_eval.{json,npz}`
(eval_brax.py) and `battery/rft_robot_<arm>_seed<k>/eval_report.json`
(upstream's `run.sh report`). Per level: success / command-conditioned
success / fall rate averaged over both passes and all episodes; every policy
ran the same levels and reset keys, so arm vs base is paired by level. 95%
CIs: 2000 bootstrap draws over seeds and levels (the base: levels only).
The criterion is `experiment.toml`'s, fixed before launch.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.tracking import tracked

ARMS = ("accel_sfl_lr1e-4", "dr_lr1e-4")
# the winner is picked among ACCEL arms; DR is the control
CANDIDATES = ("accel_sfl_lr1e-4",)
BASE = "base"
SEEDS = (0, 1, 2)
MIN_TRAVEL = 1.0
DRAWS = 2000
GUARD_DROP = 0.02
QUIET_TOL = 0.05
PRIMARY = ("random/d1.0/cmd_success_rate", "random/d1.0/success_rate")
# upstream report keys, lower is quieter
QUIET = {
    "energy_w": ("power", "mean_total"),
    "impact_accel_mps2": ("foot_force_proxy", "peak_accel_mps2"),
}


def run_name(arm: str, seed: int, kind: str) -> str:
    return f"rft_robot_{arm}_seed{seed}_{kind}_eval"


def per_level(npz: Any, prefix: str) -> dict[str, np.ndarray]:
    """(levels,) success, cmd success and fall rate over passes x episodes."""
    fell = npz[f"{prefix}_fell"]
    episodes = {
        "fell": fell,
        "lin_rel": npz[f"{prefix}_lin_rel"],
        "yaw_rel": npz[f"{prefix}_yaw_rel"],
    }
    success = (fell == 0) & (npz[f"{prefix}_travel"] >= MIN_TRAVEL)
    return {
        "success_rate": success.mean(axis=(0, 2)),
        "cmd_success_rate": tracked(episodes).mean(axis=(0, 2)),
        "fall_rate": fell.mean(axis=(0, 2)),
    }


def battery(runs_dir: Path, arm: str, seed: int) -> dict[str, float] | None:
    path = runs_dir / "battery" / f"rft_robot_{arm}_seed{seed}" / "eval_report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    out = {name: float(report[a][b]) for name, (a, b) in QUIET.items()}
    out["falls"] = float(report["termination"]["fall_count"])
    scenarios = [v for v in report["battery"].values() if isinstance(v, dict)]
    out["vibration"] = float(np.mean([s["vibration"] for s in scenarios]))
    return out


def load(runs_dir: Path, arm: str, kind: str) -> list[dict[str, Any]]:
    seeds = (0,) if arm == BASE else SEEDS
    out = []
    for seed in seeds:
        name = run_name(arm, seed, kind)
        path = runs_dir / f"{name}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        if kind == "flat":
            sets = {"flat": per_level(npz, "flat")}
        else:
            sets = {
                f"random/d{d}": per_level(npz, f"random_d{d}")
                for d in (0.25, 0.5, 0.75, 1.0)
            }
            sets["param/uniform"] = per_level(npz, "param_uniform")
        metrics = json.loads((runs_dir / f"{name}.json").read_text())
        out.append(
            {
                "seed": seed,
                "sets": sets,
                "metrics": metrics,
                "battery": battery(runs_dir, arm, seed),
            }
        )
    return out


def bootstrap(
    arm: np.ndarray, base: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """arm: (seeds, levels), base: (levels,). Mean, CI and the paired
    difference arm - base with its CI."""
    n_seeds, n_levels = arm.shape
    means, diffs = np.empty(DRAWS), np.empty(DRAWS)
    for i in range(DRAWS):
        s = rng.integers(0, n_seeds, n_seeds)
        lv = rng.integers(0, n_levels, n_levels)
        means[i] = arm[np.ix_(s, lv)].mean()
        diffs[i] = means[i] - base[lv].mean()
    return {
        "mean": float(arm.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "diff": float(arm.mean() - base.mean()),
        "diff_ci_low": float(np.percentile(diffs, 2.5)),
        "diff_ci_high": float(np.percentile(diffs, 97.5)),
    }


def level_stats(
    runs: list[dict[str, Any]], base: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_name, values in base["sets"].items():
        for metric, base_levels in values.items():
            arm = np.stack([r["sets"][set_name][metric] for r in runs])
            out[f"{set_name}/{metric}"] = {
                "pooled": bootstrap(arm, base_levels, rng),
                "per_seed": {
                    r["seed"]: bootstrap(arm[i : i + 1], base_levels, rng)
                    for i, r in enumerate(runs)
                },
            }
    return out


def beats(arm: dict[str, float], base: dict[str, float]) -> bool:
    return arm["mean"] > base["mean"] and arm["ci_low"] > base["ci_high"]


def quiet_checks(
    ours: dict[str, float] | None, base: dict[str, float] | None
) -> dict[str, bool]:
    if ours is None or base is None:
        return {"battery report present": False}
    checks = {
        f"{name} <= base x {1 + QUIET_TOL:g}": ours[name]
        <= base[name] * (1 + QUIET_TOL)
        for name in QUIET
    }
    checks["battery falls not more"] = ours["falls"] <= base["falls"]
    return checks


def verdict(
    stats: dict[str, Any],
    base_stats: dict[str, Any],
    batteries: list[dict[str, float] | None],
    base_battery: dict[str, float] | None,
    seed: Any,
) -> dict[str, Any]:
    """seed None = the 3-seed pool (quiet guards then hold for every seed)."""

    def pick(key: str) -> dict[str, float]:
        entry = stats[key]
        return entry["pooled"] if seed is None else entry["per_seed"][seed]

    primary = {key: beats(pick(key), base_stats[key]["pooled"]) for key in PRIMARY}
    guards = {
        "d0.25 success drop <= 0.02": -pick("random/d0.25/success_rate")["diff"]
        <= GUARD_DROP,
        # not worse = the paired difference is not significantly below 0
        "flat success not lower": pick("flat/success_rate")["diff_ci_high"] >= 0.0,
        "flat cmd success not lower": pick("flat/cmd_success_rate")["diff_ci_high"]
        >= 0.0,
        "flat fall rate not worse": pick("flat/fall_rate")["diff_ci_low"] <= 0.0,
        "d1.0 fall rate not worse": pick("random/d1.0/fall_rate")["diff_ci_low"] <= 0.0,
    }
    for battery_run in batteries:
        for name, ok in quiet_checks(battery_run, base_battery).items():
            guards[f"quiet: {name}"] = guards.get(f"quiet: {name}", True) and ok
    return {
        "primary": primary,
        "guards": guards,
        "win": all(primary.values()) and all(guards.values()),
    }


def arm_entry(
    runs_dir: Path, arm: str, base: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any] | None:
    final = load(runs_dir, arm, "final")
    flat = load(runs_dir, arm, "flat")
    if not final or [r["seed"] for r in final] != [r["seed"] for r in flat]:
        return None
    for r, f in zip(final, flat, strict=True):
        r["sets"].update(f["sets"])
    stats = level_stats(final, base["runs"][0], rng)
    entry: dict[str, Any] = {
        "seeds": [r["seed"] for r in final],
        "levels": stats,
        "battery": {r["seed"]: r["battery"] for r in final},
        "suite_success": {r["seed"]: r["metrics"]["suite/success_rate"] for r in final},
    }
    base_battery = base["runs"][0]["battery"]
    entry["verdict"] = {
        "pooled": verdict(
            stats, base["stats"], [r["battery"] for r in final], base_battery, None
        ),
        **{
            f"seed{r['seed']}": verdict(
                stats, base["stats"], [r["battery"]], base_battery, r["seed"]
            )
            for r in final
        },
    }
    v10box = load(runs_dir, arm, "v10box")
    if v10box and base["v10box"]:
        entry["v10box"] = level_stats(v10box, base["v10box"][0], rng)
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    base_final = load(args.runs_dir, BASE, "final")
    base_flat = load(args.runs_dir, BASE, "flat")
    if not base_final or not base_flat:
        raise FileNotFoundError(
            f"{args.runs_dir}: no {run_name(BASE, 0, 'final')}.npz or "
            f"{run_name(BASE, 0, 'flat')}.npz; copy $PDW/runs/robot_ft here first"
        )
    base_final[0]["sets"].update(base_flat[0]["sets"])
    base = {
        "runs": base_final,
        "stats": level_stats(base_final, base_final[0], rng),
        "v10box": load(args.runs_dir, BASE, "v10box"),
    }
    results: dict[str, Any] = {
        BASE: {
            "levels": base["stats"],
            "battery": base_final[0]["battery"],
            "suite_success": base_final[0]["metrics"]["suite/success_rate"],
        }
    }
    for arm in ARMS:
        entry = arm_entry(args.runs_dir, arm, base, rng)
        if entry is not None:
            entry["candidate"] = arm in CANDIDATES
            results[arm] = entry
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    for arm in ARMS:
        if arm in results:
            wins = {k: v["win"] for k, v in results[arm]["verdict"].items()}
            print(arm, wins)


if __name__ == "__main__":
    main()
