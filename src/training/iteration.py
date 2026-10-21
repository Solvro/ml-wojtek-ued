from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.training import types

from config.schema import Score
from hfield.level_space import mean_abs_height
from robots.scale import level_steps, min_travel
from training.context import TrainContext
from training.student import Student


class LaneTotals:
    """Per-env sums over one iteration's unrolls."""

    def __init__(self, n: int) -> None:
        self.regret = jnp.zeros(n)
        self.episode_return = jnp.zeros(n)
        self.fell_any = jnp.zeros(n)
        self.leaked = jnp.zeros(n, bool)
        self.bc_kl = jnp.zeros(())
        self.logs: list[dict[str, jax.Array]] = []

    def add(
        self,
        data: types.Transition,
        leaked: jax.Array,
        regret: jax.Array,
        bc_kl: jax.Array | None,
    ) -> None:
        log = data.extras["log"]
        self.regret = self.regret + regret
        if bc_kl is not None:
            self.bc_kl = self.bc_kl + bc_kl
        self.episode_return = self.episode_return + data.reward.sum(axis=1)
        self.fell_any = jnp.maximum(self.fell_any, log["fell"].max(axis=1))
        self.leaked = self.leaked | leaked.any(axis=1)
        self.logs.append(jax.tree.map(jnp.mean, log))


def run_levels(
    ctx: TrainContext,
    levels: np.ndarray,
    train: bool,
    *,
    student: Student | None = None,
    teacher: Student | None = None,
    bc_coef: float | None = None,
) -> dict[str, Any]:
    """One outer iteration, level i on tile i (one lane each); a PPO update
    after every unroll when training. With a `teacher` (PAIRED), `bc_kl` =
    mean KL(teacher || student) over the observations; a `bc_coef` (0 on
    off-iterations) also adds `bc_coef` x that KL to every update."""
    env, cfg = ctx.env, ctx.cfg
    student = student or ctx.student
    ref = teacher.as_teacher() if teacher else None
    loss_ref = None if bc_coef is None else ref
    key = ctx.next_key()
    rendered = ctx.space.render(levels)
    hfield = ctx.assemble(jnp.asarray(rendered))
    tile_xy = ctx.layout.tile_xy(cfg.num_lanes, cfg.envs_per_lane)
    n = cfg.num_lanes * cfg.envs_per_lane
    dr_model, _ = env.sample_dr(ctx.next_key(), n)
    _, state = ctx.reset(jax.random.split(key, n), hfield, dr_model, tile_xy)
    start_xy = state.data.qpos[:, :2]
    totals = LaneTotals(n)
    unrolls = level_steps(cfg.robot, cfg.level_steps) // cfg.ppo.unroll_length
    for _ in range(unrolls):
        state, data = ctx.unroll(
            student.inference_params(),
            state,
            ctx.next_key(),
            hfield,
            dr_model,
            deterministic=False,
            length=cfg.ppo.unroll_length,
        )
        if train:
            student.train_step(data, ctx.next_key(), loss_ref, bc_coef)
        bc_kl = None
        if ref is not None:
            bc_kl = student.bc_kl(
                student.params.policy,
                student.normalizer,
                ref,
                data.observation,
            )
        regret = student.regret(student.params, student.normalizer, data)
        xy = data.extras["log"]["xy"]
        leaked = ctx.layout.leaked(xy, tile_xy[:, None], env.travel.sensing)
        totals.add(data, leaked, regret, bc_kl)
    travel = jnp.linalg.norm(state.data.qpos[:, :2] - start_xy, axis=-1)
    out = summarize_lanes(ctx, totals, travel, unrolls)
    out["lane_roughness"] = mean_abs_height(ctx.space.config, rendered)
    out["levels"] = levels
    if ref is not None:
        out["bc_kl"] = float(totals.bc_kl) / unrolls
    return out


def summarize_lanes(
    ctx: TrainContext, totals: LaneTotals, travel: jax.Array, unrolls: int
) -> dict[str, Any]:
    cfg = ctx.cfg
    lanes = cfg.num_lanes
    lane_regret = np.asarray(totals.regret.reshape(lanes, -1).mean(axis=1))
    lane_returns = np.asarray(totals.episode_return.reshape(lanes, -1))
    threshold = min_travel(cfg.robot, cfg.min_travel)
    success = np.asarray(
        (totals.fell_any == 0) & (travel >= threshold), dtype=float
    )
    lane_success = success.reshape(lanes, -1).mean(axis=1)
    lane_regret = lane_regret / unrolls
    # SFL learnability: highest where the student sometimes succeeds
    learnability = lane_success * (1.0 - lane_success)
    log = jax.tree.map(lambda *x: float(np.mean(x)), *totals.logs)
    use_learnability = cfg.mode.score == Score.learnability
    return {
        "regret": lane_regret,
        "score": learnability if use_learnability else lane_regret,
        "lane_returns": lane_returns,
        "train_success": float(success.mean()),
        "learnability_mean": float(learnability.mean()),
        "falls_per_step": log["fell"],
        "diverged_per_step": log["diverged"],
        "reward_per_step": log["reward"],
        "cmd_vx": log["cmd_vx"],
        "forward_vel": log["forward_vel"],
        "travel": float(travel.mean()),
        # envs whose feet or height scan reached another lane's level
        "tile_leak_frac": float(totals.leaked.mean()),
    }
