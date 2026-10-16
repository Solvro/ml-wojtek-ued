"""Final-eval tables and the win verdict of deploy-accel-ft.

    PYTHONPATH=src uv run python experiments/deploy-accel-ft/analyze.py \
        --runs_dir <local copy of $PDW/runs/deploy_accel_ft> --out results.json

Reads every `dft2_deploy_<arm>_seed<k>_{final,v10box}_eval.{json,npz}`
(eval_brax.py output). Per level: success / command-conditioned success
averaged over both passes and all episodes; every policy ran the same
levels and reset keys, so arm vs base is paired by level. 95% CIs: 2000
bootstrap draws resampling seeds and levels (the base: levels only).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils.tracking import tracked

ARMS = (
    "accel",
    "accel_lr1e-4",
    "dr",
    "accel_hard_lr1e-4",
    "dr_lr1e-4",
    "accel_sfl_lr1e-4",
    "accel_lr1e-4_1b",
)
BASE = "base"
SEEDS = (0, 1, 2)
MIN_TRAVEL = 1.0
DRAWS = 2000
GUARD_DROP = 0.02


def run_name(arm: str, seed: int, kind: str) -> str:
    return f"dft2_deploy_{arm}_seed{seed}_{kind}_eval"


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


def load(runs_dir: Path, arm: str, kind: str) -> list[dict[str, Any]]:
    seeds = (0,) if arm == BASE else SEEDS
    out = []
    for seed in seeds:
        name = run_name(arm, seed, kind)
        path = runs_dir / f"{name}.npz"
        if not path.exists():
            continue
        npz = np.load(path)
        sets = {
            f"random/d{d}": per_level(npz, f"random_d{d}")
            for d in (0.25, 0.5, 0.75, 1.0)
        }
        sets["param/uniform"] = per_level(npz, "param_uniform")
        if kind == "final":
            flat = runs_dir / f"{run_name(arm, seed, 'flat')}.npz"
            if not flat.exists():
                raise FileNotFoundError(
                    f"{flat} missing: the flat guard needs the 1024-level flat "
                    "eval (eval.sbatch, FLAT_ONLY=1 for an evaluated policy)"
                )
            sets["flat"] = per_level(np.load(flat), "flat")
        metrics = json.loads((runs_dir / f"{name}.json").read_text())
        out.append({"seed": seed, "sets": sets, "metrics": metrics})
    return out


def bootstrap(
    arm: np.ndarray, base: np.ndarray | None, rng: np.random.Generator
) -> dict[str, float]:
    """arm: (seeds, levels). Mean and CI; with base (levels,), the paired
    difference arm - base too."""
    n_seeds, n_levels = arm.shape
    means, diffs = np.empty(DRAWS), np.empty(DRAWS)
    for i in range(DRAWS):
        s = rng.integers(0, n_seeds, n_seeds)
        lv = rng.integers(0, n_levels, n_levels)
        means[i] = arm[np.ix_(s, lv)].mean()
        if base is not None:
            diffs[i] = means[i] - base[lv].mean()
    out = {
        "mean": float(arm.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }
    if base is not None:
        out["diff"] = float(arm.mean() - base.mean())
        out["diff_ci_low"] = float(np.percentile(diffs, 2.5))
        out["diff_ci_high"] = float(np.percentile(diffs, 97.5))
    return out


def level_stats(
    runs: list[dict[str, Any]], base: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for set_name, values in base["sets"].items():
        for metric, base_levels in values.items():
            key = f"{set_name}/{metric}"
            arm = np.stack([r["sets"][set_name][metric] for r in runs])
            out[key] = {
                "pooled": bootstrap(arm, base_levels, rng),
                "per_seed": {
                    r["seed"]: bootstrap(arm[i : i + 1], base_levels, rng)
                    for i, r in enumerate(runs)
                },
            }
    return out


def json_stats(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out = {}
    for key in keys:
        values = [r["metrics"][key] for r in runs if key in r["metrics"]]
        out[key] = {
            "mean": float(np.mean(values)) if values else None,
            "per_seed": {r["seed"]: r["metrics"].get(key) for r in runs},
        }
    return out


PRIMARY = ("random/d1.0/cmd_success_rate", "random/d1.0/success_rate")


def beats(arm: dict[str, float], base: dict[str, float]) -> bool:
    return arm["mean"] > base["mean"] and arm["ci_low"] > base["ci_high"]


def verdict(native: dict[str, Any], base: dict[str, Any], seed: Any) -> dict[str, Any]:
    """seed None = the 3-seed pool."""

    def pick(key: str) -> dict[str, float]:
        stats = native[key]
        return stats["pooled"] if seed is None else stats["per_seed"][seed]

    primary = {key: beats(pick(key), base[key]["pooled"]) for key in PRIMARY}
    d025 = pick("random/d0.25/success_rate")
    fall = pick("random/d1.0/fall_rate")
    flat = pick("flat/success_rate")
    guards = {
        "d0.25 success drop <= 0.02": -d025["diff"] <= GUARD_DROP,
        "flat success drop <= 0.02": -flat["diff"] <= GUARD_DROP,
        # not worse = the paired difference is not significantly above 0
        "d1.0 fall rate not worse": fall["diff_ci_low"] <= 0.0,
    }
    return {
        "primary": primary,
        "guards": guards,
        "win": all(primary.values()) and all(guards.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs_dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    results: dict[str, Any] = {}
    for kind in ("final", "v10box"):
        base_runs = load(args.runs_dir, BASE, kind)
        if not base_runs:
            raise FileNotFoundError(
                f"{args.runs_dir}: no {run_name(BASE, 0, kind)}.npz; copy "
                "$PDW/runs/deploy_accel_ft here first"
            )
        base = base_runs[0]
        base_stats = level_stats(base_runs, base, rng)
        json_keys = ["suite/success_rate", "suite/cmd_success_rate"]
        json_keys += ["flat/success_rate", "flat/fall_rate"]
        section: dict[str, Any] = {
            BASE: {"levels": base_stats, "json": json_stats(base_runs, json_keys)}
        }
        for arm in ARMS:
            runs = load(args.runs_dir, arm, kind)
            if not runs:
                continue
            stats = level_stats(runs, base, rng)
            js = json_stats(runs, json_keys)
            entry: dict[str, Any] = {
                "seeds": [r["seed"] for r in runs],
                "levels": stats,
                "json": js,
            }
            if kind == "final":
                entry["verdict"] = {
                    "pooled": verdict(stats, base_stats, None),
                    **{
                        f"seed{r['seed']}": verdict(stats, base_stats, r["seed"])
                        for r in runs
                    },
                }
            section[arm] = entry
        results[kind] = section
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
