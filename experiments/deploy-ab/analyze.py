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


def print_upstream(name: str) -> None:
    """A1 runs: w01-tek train.py logs brax ppo.train's eval/* (128 envs,
    1000-step episodes, stochastic policy) and training episode/* metrics.
    eval/episode_reward/<term> is the raw term summed over an episode; the
    termination term is zeroed inside the 1 s spawn grace, so fall ends are
    read from base_contact_at_done (1 when the episode ended base-down)."""
    api = wandb.Api(timeout=120)
    runs = [
        r
        for r in api.runs(PROJECT, filters={"display_name": name})
        if "oom" not in r.tags
    ]
    if not runs:
        print(f"\n{name}: no run")
        return
    run = max(runs, key=lambda r: r.created_at)
    scales, dt = reward_weights(DEPLOY)
    terms = [f"eval/episode_reward/{k}" for k in scales]
    keys = [
        "eval/episode_reward",
        "eval/avg_episode_length",
        "eval/episode_base_contact_at_done",
        "eval/episode_no_progress_cut",
        "perf/steps_per_sec",
    ]
    rows = run.history(keys=keys + terms, pandas=False, samples=1000)
    train = run.history(keys=["episode/length"], pandas=False, samples=1000)
    print(f"\n### {name} ({run.state})\n")
    print(
        "| env steps | eval ep_len | eval ends base-down | eval no-progress cut "
        "| eval reward | "
        "train ep_len | steps/s | top-3 weighted terms / step |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        step = row["_step"]
        ep_len = row["eval/avg_episode_length"]
        weighted = {
            k: row[f"eval/episode_reward/{k}"] / ep_len * scale * dt
            for k, scale in scales.items()
            if scale and row.get(f"eval/episode_reward/{k}") is not None
        }
        top = sorted(weighted.items(), key=lambda kv: -abs(kv[1]))[:3]
        near = [t for t in train if abs(t["_step"] - step) <= 3e6]
        train_len = near[-1]["episode/length"] if near else None
        print(
            f"| {step / 1e6:.1f}M | {ep_len:.1f} | "
            f"{fmt(row.get('eval/episode_base_contact_at_done'))} | "
            f"{fmt(row.get('eval/episode_no_progress_cut'))} | "
            f"{row['eval/episode_reward']:.3g} | {fmt(train_len)} | "
            f"{fmt(row.get('perf/steps_per_sec'))} | "
            + ", ".join(f"{k} {v:+.3f}" for k, v in top)
            + " |"
        )


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
    for name in ("deploy_ab_a1_upstream_seed0", "deploy_ab_a1long_upstream_seed0"):
        print_upstream(name)
    print("\n```json")
    print(json.dumps({arm: len(rows) for arm, rows in cache.items()}))
    print("```")


if __name__ == "__main__":
    main()
