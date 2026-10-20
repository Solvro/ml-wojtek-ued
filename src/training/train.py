import time
from pathlib import Path
from typing import Any

import numpy as np
import wandb
from brax.io import model as brax_model

from config.schema import TrainConfig, as_dict
from training.buffer import LevelBuffer
from training.build import build, check_config
from training.context import TrainContext
from training.curricula import PAIRED, run_iteration
from training.evaluate import evaluate
from training.metrics import level_composition
from training.summary import SummaryWindow, TrainingSummary, write_summary

EVAL_SEED = 1234


class StepCounter:
    def __init__(self, cfg: TrainConfig) -> None:
        self.per_iteration = cfg.num_lanes * cfg.envs_per_lane * cfg.level_steps
        # the PAIRED antagonist simulates (and trains on) the same levels
        self.sims_per_iteration = 2 if cfg.mode.curriculum in PAIRED else 1
        self.env_steps = 0
        self.sim_steps = 0
        self.student_train_steps = 0

    def add(self, replay: int) -> None:
        self.env_steps += self.per_iteration
        self.sim_steps += self.sims_per_iteration * self.per_iteration
        # replay == 1 exactly when the student's PPO updated on this data
        self.student_train_steps += replay * self.per_iteration


def iteration_log(
    ctx: TrainContext, out: dict[str, Any], buffer: LevelBuffer
) -> dict[str, Any]:
    """The iteration's wandb row, minus the step and timing counters."""
    regret = np.asarray(out.pop("regret"))
    out["level_roughness_m"] = float(np.mean(out.pop("lane_roughness")))
    out.update(level_composition(ctx.space, out.pop("levels")))
    for key in ("lane_returns", "score"):
        out.pop(key)
    scores = buffer.scores
    return {
        **out,
        "regret_mean": float(regret.mean()),
        "regret_max": float(regret.max()),
        "buffer_size": len(buffer),
        "buffer_score_mean": float(scores.mean()) if len(buffer) else 0.0,
    }


def save_checkpoint(ctx: TrainContext, summary: TrainingSummary | None) -> None:
    cfg = ctx.cfg
    if not cfg.checkpoint_dir:
        return
    path = Path(cfg.checkpoint_dir) / f"{cfg.name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    brax_model.save_params(str(path), ctx.student.inference_params())
    if summary is not None:
        write_summary(path, summary)


def train(cfg: TrainConfig, mode_name: str) -> None:
    check_config(cfg)
    launch = time.time()
    run = wandb.init(
        entity=cfg.entity or None,
        project=cfg.project,
        name=cfg.name,
        tags=["brax-ued", mode_name, cfg.robot.name, *cfg.tags],
        config=as_dict(cfg),
    )
    ctx = build(cfg)
    buffer = LevelBuffer(cfg.mode.buffer_size)
    steps = StepCounter(cfg)
    it, start = 0, time.time()
    window = SummaryWindow()
    while steps.env_steps < cfg.num_timesteps:
        out = run_iteration(ctx, buffer, it)
        steps.add(out["replay"])
        row = iteration_log(ctx, out, buffer)
        window.add(row)
        log = {
            "env_steps": steps.env_steps,
            "iteration": it,
            **row,
            "steps_per_sec": steps.env_steps / (time.time() - start),
            "sim_steps": steps.sim_steps,
            "student_train_steps": steps.student_train_steps,
            "elapsed_s": time.time() - launch,
        }
        last = steps.env_steps >= cfg.num_timesteps
        if it % cfg.eval_every == 0 or last:
            log.update(evaluate(ctx, np.random.default_rng(EVAL_SEED)))
        run.log(log)
        rounded = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in log.items()
        }
        print(rounded, flush=True)
        it += 1
    summary = window.summary(steps.env_steps, time.time() - launch)
    save_checkpoint(ctx, summary)
    run.summary["final/elapsed_h"] = (time.time() - launch) / 3600
    run.summary["final/sim_steps"] = steps.sim_steps
    run.summary["final/student_train_steps"] = steps.student_train_steps
    run.finish()
