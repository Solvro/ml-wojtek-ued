"""Tables for report.md from the wandb runs of experiments/deploy-ab.

PYTHONPATH=src uv run python experiments/deploy-ab/analyze.py > tables.md
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from scripts.brax_ued import parse_args, run_spec
from wojtek_env.envs import registry

PROJECT = "dominikgalus9-wroc-aw-university-of-science-and-technology/wojtyla"
ARMS_DIR = Path(__file__).parent / "arms"
CHECKPOINTS = (10e6, 25e6, 50e6)
BASE = ["--mode", "accel", "--name", "t", "--level_space", "param"]
DEPLOY = [*BASE, "--profile", "deploy"]
# arm -> (wandb run, brax_ued flags that set its reward config)
RUNS: dict[str, tuple[str, list[str]]] = {
    "v10 seed4 (benchmark)": ("v10_wojtek_accel_seed4", BASE),
    "A0 deploy (T4 seed0)": ("deploy_wojtek_accel_seed0", DEPLOY),
    "A2 v10 reward": (
        "deploy_ab_a2_v10_reward_seed0",
        [*DEPLOY, "--env_override", str(ARMS_DIR / "a2_v10_reward.json")],
    ),
    "A3 level_steps 1000": ("deploy_ab_a3_level1000_seed0", DEPLOY),
    "A4 clip 0.3": ("deploy_ab_a4_clip03_seed0", DEPLOY),
    "A5 flat levels": ("deploy_ab_a5_flat_seed0", DEPLOY),
    "A6 torque_rate -1e-4": (
        "deploy_ab_a6_torque_rate_seed0",
        [*DEPLOY, "--env_override", str(ARMS_DIR / "a6_torque_rate.json")],
    ),
}
TRAIN = ["falls_per_step", "train_success", "reward_per_step", "steps_per_sec"]
EVAL = [
    f"eval/d{d}/{m}"
    for d in ("0.25", "1.0")
    for m in ("success_rate", "fall_rate", "mean_travel")
]


def reward_weights(flags: list[str]) -> tuple[dict[str, float], float]:
    config = registry.TASKS["joystick"][1]()
    registry._apply_overrides(config, run_spec(parse_args(flags))["env_overrides"])
    return dict(config.reward.scales), float(config.ctrl_dt)


def history(name: str) -> list[dict[str, Any]]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(PROJECT, filters={"display_name": name}))
    if not runs:
        return []
    run = max(runs, key=lambda r: r.created_at)
    return [row for row in run.scan_history() if row.get("env_steps") is not None]


def at(rows: list[dict[str, Any]], key: str, step: float) -> float | None:
    """Value of `key` at the row nearest to `step` that carries it, if one
    lies within 35% of `step` (evals are sparser than iterations)."""
    have = [r for r in rows if r.get(key) is not None]
    if not have:
        return None
    row = min(have, key=lambda r: abs(r["env_steps"] - step))
    if abs(row["env_steps"] - step) > 0.35 * step:
        return None
    return float(row[key])


def top_terms(
    rows: list[dict[str, Any]], flags: list[str], step: float, n: int = 3
) -> str:
    scales, dt = reward_weights(flags)
    terms = {}
    for name, scale in scales.items():
        value = at(rows, f"reward_term/{name}", step)
        if value is not None and scale:
            terms[name] = value * scale * dt
    if not terms:
        return "-"
    worst = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:n]
    return ", ".join(f"{k} {v:+.3f}" for k, v in worst)


def fmt(value: float | None) -> str:
    return "-" if value is None or np.isnan(value) else f"{value:.3g}"


def main() -> None:
    cache: dict[str, list[dict[str, Any]]] = {}
    for arm, (name, _) in RUNS.items():
        cache[arm] = history(name)
    for step in CHECKPOINTS:
        print(f"\n### {step / 1e6:.0f}M env steps\n")
        print(
            "| arm | "
            + " | ".join(k.removeprefix("eval/") for k in TRAIN + EVAL)
            + " | top-3 weighted reward terms / step |"
        )
        print("|---" * (len(TRAIN) + len(EVAL) + 2) + "|")
        for arm, (_, flags) in RUNS.items():
            rows = cache[arm]
            cells = [fmt(at(rows, k, step)) for k in TRAIN + EVAL]
            print(
                f"| {arm} | "
                + " | ".join(cells)
                + f" | {top_terms(rows, flags, step)} |"
            )
    print("\n```json")
    print(json.dumps({arm: len(rows) for arm, rows in cache.items()}))
    print("```")


if __name__ == "__main__":
    main()
