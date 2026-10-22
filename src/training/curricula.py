from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from config.schema import Curriculum, Generator, Mutation, Score
from training.buffer import LevelBuffer
from training.context import TrainContext
from training.iteration import run_levels
from training.paired import bc_coefs, paired_regret
from training.progress import learning_progress, unvisited, update_progress
from training.terrain_cl import LaneOutcome

# curricula that train a protagonist and an antagonist
PAIRED = (Curriculum.paired, Curriculum.paired_evo)
# curricula on ACCEL's level buffer (`accel_iteration`)
BUFFERED = (Curriculum.accel, Curriculum.plr, Curriculum.paired_evo)


def easy_levels(ctx: TrainContext, n: int) -> np.ndarray:
    return ctx.space.easy(ctx.rng, n, ctx.cfg.mode.generator_scale)


def dr_levels(ctx: TrainContext, curriculum: Curriculum) -> np.ndarray:
    n, rng, space = ctx.cfg.num_lanes, ctx.rng, ctx.space
    if curriculum == Curriculum.dr_easy:
        return easy_levels(ctx, n)
    if curriculum == Curriculum.dr_mixed:
        # a per-level difficulty in [0, 1]: covers the easy levels too
        return np.concatenate(
            [space.scaled(rng, 1, d) for d in rng.uniform(0.0, 1.0, n)]
        )
    return space.uniform(rng, n)


def mutate(ctx: TrainContext, level: np.ndarray) -> np.ndarray:
    mode = ctx.cfg.mode
    # symmetric local edits smooth as often as they roughen; "grow" also
    # scales the whole genome away from flat so each generation has a direction
    grow_max = mode.grow_max if mode.mutation == Mutation.grow else None
    return ctx.space.mutate(level, ctx.rng, grow_max)


def run_pair(
    ctx: TrainContext, levels: np.ndarray, train: bool, it: int
) -> dict[str, Any]:
    """Both PAIRED students on `levels`; score = PAIRED regret per level.
    The antagonist runs second, so BiBC distils it from the protagonist this
    iteration's updates produced."""
    prot, antag = ctx.student, ctx.antagonist
    prot_coef, antag_coef = bc_coefs(ctx.cfg.mode, it)
    out = run_levels(ctx, levels, train, teacher=antag, bc_coef=prot_coef)
    antagonist = run_levels(
        ctx, levels, train, student=antag, teacher=prot, bc_coef=antag_coef
    )
    regret = paired_regret(out["lane_returns"], antagonist["lane_returns"])
    prot_bc_kl = out.pop("bc_kl")
    return {
        **out,
        "score": regret,
        "paired_regret_mean": float(regret.mean()),
        "prot_return": float(out["lane_returns"].mean()),
        "antag_return": float(antagonist["lane_returns"].mean()),
        "antag_forward_vel": antagonist["forward_vel"],
        "prot_bc_kl": prot_bc_kl,
        "antag_bc_kl": antagonist["bc_kl"],
        "bc_coef": prot_coef if train and prot_coef is not None else 0.0,
    }


def score_levels(
    ctx: TrainContext,
    levels: np.ndarray,
    train: bool,
    it: int,
    prior: np.ndarray,
) -> dict[str, Any]:
    """`prior`: the levels' learning-progress state before this visit;
    `progress` comes back with the visit folded in."""
    mode = ctx.cfg.mode
    if mode.curriculum == Curriculum.paired_evo:
        out = run_pair(ctx, levels, train, it)
    else:
        out = run_levels(ctx, levels, train)
    out["progress"] = update_progress(prior, out["lane_success"], mode)
    if mode.score == Score.lp:
        out["score"] = learning_progress(out["progress"])
    return out


def explore_levels(ctx: TrainContext, fresh: int) -> np.ndarray:
    mode = ctx.cfg.mode
    if mode.curriculum == Curriculum.plr:
        return ctx.space.uniform(ctx.rng, fresh)
    if mode.generator == Generator.easy:
        return easy_levels(ctx, fresh)
    return dr_levels(ctx, Curriculum.dr_mixed)[:fresh]


def accel_iteration(
    ctx: TrainContext, buffer: LevelBuffer, it: int
) -> dict[str, Any]:
    """Replay (train on buffer levels, queue their mutations) or explore
    (score the queued children plus fresh levels, no training)."""
    cfg, rng = ctx.cfg, ctx.rng
    full = len(buffer) >= cfg.num_lanes
    if full and rng.random() < cfg.mode.replay_prob:
        idx = buffer.sample(rng, cfg.num_lanes, it)
        levels = np.stack([buffer.levels[i] for i in idx])
        out = score_levels(ctx, levels, True, it, buffer.progress[idx])
        progress = out.pop("progress")
        buffer.update(idx, out["score"], it, progress)
        mutates = (Curriculum.accel, Curriculum.paired_evo)
        if cfg.mode.curriculum in mutates:
            ctx.children = np.stack([mutate(ctx, level) for level in levels])
            ctx.children_progress = progress
        return {**out, "replay": 1, **buffer.stats()}
    new = explore_levels(ctx, cfg.num_lanes - len(ctx.children))
    queued = len(ctx.children)
    levels = np.concatenate([ctx.children, new]) if queued else new
    prior = np.concatenate([ctx.children_progress, unvisited(len(new))])
    ctx.children = np.zeros((0, *ctx.space.shape))
    ctx.children_progress = unvisited(0)
    out = score_levels(ctx, levels, False, it, prior)
    progress = out.pop("progress")
    buffer.add(levels, out["score"], it, out["lane_roughness"], progress)
    return {**out, "replay": 0, **buffer.stats()}


def terrain_iteration(ctx: TrainContext) -> dict[str, Any]:
    lanes = ctx.terrain
    if lanes is None:
        raise ValueError("terrain_iteration needs mode.curriculum=terrain_cl")
    out = run_levels(ctx, lanes.levels(), train=True)
    outcome = LaneOutcome(
        out["lane_success"], out["lane_travel"], out["lane_cmd_distance"]
    )
    return {**out, **lanes.advance(outcome, ctx.rng), "replay": 1}


def paired_iteration(ctx: TrainContext, it: int) -> dict[str, Any]:
    adversary = ctx.adversary
    if adversary is None:
        raise ValueError("paired_iteration needs mode.curriculum=paired")
    lanes = ctx.cfg.num_lanes
    z = jax.random.normal(ctx.next_key(), (lanes, adversary.latent_dim))
    keys = jax.random.split(ctx.next_key(), lanes)
    theta, log_prob = adversary.sample(adversary.generator, z, keys)
    levels = ctx.space.from_generated(np.asarray(theta), ctx.rng)
    out = run_pair(ctx, levels, True, it)
    regret = jnp.asarray(out["score"])
    gen_loss = adversary.update(z, theta, log_prob, regret)
    return {**out, "replay": 1, "gen_loss": gen_loss}


def run_iteration(
    ctx: TrainContext, buffer: LevelBuffer, it: int
) -> dict[str, Any]:
    curriculum = ctx.cfg.mode.curriculum
    if curriculum in BUFFERED:
        return accel_iteration(ctx, buffer, it)
    if curriculum == Curriculum.paired:
        return paired_iteration(ctx, it)
    if curriculum == Curriculum.terrain_cl:
        return terrain_iteration(ctx)
    levels = dr_levels(ctx, curriculum)
    return {**run_levels(ctx, levels, train=True), "replay": 1}
