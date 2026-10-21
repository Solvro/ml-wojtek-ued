from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import EvalConfig
from env.factory import tile_reset
from evaluation.rollout import (
    EvalContext,
    Passes,
    level_success,
    load_params,
    make_eval_env,
    two_passes,
)
from evaluation.stats import bootstrap_ci, cross_fit_cvar, envelope
from evaluation.suite import SUITE_STRENGTHS, suite_levels
from hfield.level_space import PARAMS, LevelSpace, ParamSpace, make_level_space
from robots.scale import min_travel, robot_scale
from training.ppo import make_network
from training.unroll import make_unroll

DIFFICULTIES = (0.25, 0.5, 0.75, 1.0)
# capability axes: (name, genome index, genome value at full strength); the
# axis runs from the flat genome to that end with every other feature off
AXES = (
    ("rough", 0, 1.0),
    ("slope_up", 2, 1.0),
    ("slope_down", 2, 0.0),
    ("stairs_up", 3, 1.0),
    ("stairs_down", 3, 0.0),
    ("gaps", 5, 1.0),
    ("blocks", 6, 1.0),
)
AXIS_STEPS = 6

Metrics = dict[str, float]
Raw = dict[str, Any]


def ci_into(
    out: Metrics,
    tag: str,
    statistic: Callable[[np.ndarray], float],
    n: int,
    rng: np.random.Generator,
) -> None:
    out[f"{tag}_ci_low"], out[f"{tag}_ci_high"] = bootstrap_ci(
        statistic, n, rng
    )


def summarize(prefix: str, passes: Passes, cfg: EvalConfig) -> Metrics:
    """Success, falls, travel, cross-fitted CVaR and bootstrap CIs."""
    (fell_a, travel_a), (fell_b, travel_b) = passes
    threshold = min_travel(cfg.robot, cfg.min_travel)
    success_a = level_success(fell_a, travel_a, threshold)
    success_b = level_success(fell_b, travel_b, threshold)
    rng = np.random.default_rng(0)
    n = len(success_a)
    mean = 0.5 * (success_a + success_b)
    out = {
        f"{prefix}/success_rate": float(mean.mean()),
        f"{prefix}/fall_rate": float(0.5 * (fell_a.mean() + fell_b.mean())),
        f"{prefix}/mean_travel": float(
            0.5 * (travel_a.mean() + travel_b.mean())
        ),
    }
    ci_into(
        out, f"{prefix}/success_rate", lambda i: float(mean[i].mean()), n, rng
    )
    for alpha in cfg.alphas:
        tag = f"{prefix}/cvar{alpha}_success"
        out[tag] = float(cross_fit_cvar(success_a, success_b, alpha))

        def cvar(i: np.ndarray, alpha: float = alpha) -> float:
            return cross_fit_cvar(success_a[i], success_b[i], alpha)

        ci_into(out, tag, cvar, n, rng)
    if success_a.std() > 0 and success_b.std() > 0:
        out[f"{prefix}/level_success_corr"] = float(
            np.corrcoef(success_a, success_b)[0, 1]
        )
    return out


def stacked(passes: Passes, prefix: str) -> Raw:
    return {
        f"{prefix}_fell": np.stack([p[0] for p in passes]),
        f"{prefix}_travel": np.stack([p[1] for p in passes]),
    }


def axis_ramp(
    ctx: EvalContext[EvalConfig],
    space: ParamSpace,
    axis: tuple[str, int, float],
    rng: np.random.Generator,
    key: jax.Array,
) -> tuple[Metrics, jax.Array]:
    """One capability envelope: the feature ramped from off to full."""
    cfg = ctx.cfg
    name, index, end = axis
    param = PARAMS[index]
    physical, success = [], []
    for t in np.linspace(0.0, 1.0, AXIS_STEPS + 1)[1:]:
        genome = np.tile(space.flat, (cfg.suite_instances, 1))
        genome[:, index] = space.flat[index] + (end - space.flat[index]) * t
        levels = space.from_generated(genome, rng)
        key, sub = jax.random.split(key)
        passes = two_passes(ctx, space.render(levels), sub)
        summary = summarize(f"axis/{name}", passes, cfg)
        value = param.low + genome[0, index] * (param.high - param.low)
        physical.append(abs(value))
        success.append(summary[f"axis/{name}/success_rate"])
    physical, success = np.asarray(physical), np.asarray(success)
    metrics = {}
    for j, (v, sr) in enumerate(zip(physical, success, strict=True)):
        metrics[f"axis/{name}/value{j}"] = float(v)
        metrics[f"axis/{name}/success{j}"] = float(sr)
    metrics[f"axis/{name}/envelope"] = envelope(physical, success)
    metrics[f"axis/{name}/auc"] = float(success.mean())
    print(
        name,
        param.name,
        np.round(physical, 3),
        np.round(success, 2),
        flush=True,
    )
    return metrics, key


def param_space_sets(
    ctx: EvalContext[EvalConfig], space: ParamSpace, key: jax.Array
) -> tuple[Metrics, Raw]:
    """`param/uniform` and the `axis/<feature>` capability envelopes."""
    cfg = ctx.cfg
    rng = np.random.default_rng(cfg.seed + 1)
    key, sub = jax.random.split(key)
    uniform = space.uniform(rng, cfg.num_levels)
    passes = two_passes(ctx, space.render(uniform), sub)
    metrics = summarize("param/uniform", passes, cfg)
    raw = {**stacked(passes, "param_uniform"), "param_uniform_levels": uniform}
    for axis in AXES:
        axis_metrics, key = axis_ramp(ctx, space, axis, rng, key)
        metrics.update(axis_metrics)
    return metrics, raw


def random_sets(
    ctx: EvalContext[EvalConfig], space: LevelSpace, key: jax.Array
) -> tuple[Metrics, Raw, jax.Array]:
    """`random/d<d>`: the same base levels, shrunk toward flat by d."""
    cfg = ctx.cfg
    metrics: Metrics = {}
    raw: Raw = {
        "random_levels": space.scaled(
            np.random.default_rng(cfg.seed), cfg.num_levels, 1.0
        )
    }
    for difficulty in DIFFICULTIES:
        levels = space.scaled(
            np.random.default_rng(cfg.seed), cfg.num_levels, difficulty
        )
        key, sub = jax.random.split(key)
        passes = two_passes(ctx, space.render(levels), sub)
        metrics.update(summarize(f"random/d{difficulty}", passes, cfg))
        raw.update(stacked(passes, f"random_d{difficulty}"))
        print(
            {
                k: round(v, 3)
                for k, v in metrics.items()
                if f"d{difficulty}/" in k
            },
            flush=True,
        )
    return metrics, raw, key


def suite_sets(
    ctx: EvalContext[EvalConfig], rng: np.random.Generator, key: jax.Array
) -> tuple[Metrics, Raw, jax.Array]:
    """The held-out structured suite: success per cell, mean and CVaR."""
    cfg = ctx.cfg
    metrics: Metrics = {}
    cells = suite_levels(
        ctx.config,
        SUITE_STRENGTHS,
        cfg.suite_instances,
        rng,
        robot_scale(cfg.robot),
    )
    suite_success = []
    for name, hfields in cells:
        key, sub = jax.random.split(key)
        summary = summarize(name, two_passes(ctx, hfields, sub), cfg)
        metrics[f"{name}/success_rate"] = summary[f"{name}/success_rate"]
        suite_success.append(summary[f"{name}/success_rate"])
    suite_success = np.asarray(suite_success)
    metrics["suite/success_rate"] = float(suite_success.mean())
    for alpha in cfg.alphas:
        k = max(1, int(np.ceil(alpha * len(suite_success))))
        worst = np.sort(suite_success)[:k]
        metrics[f"suite/cvar{alpha}_success"] = float(worst.mean())
    return metrics, {"suite_success": suite_success}, key


def evaluate(ctx: EvalContext[EvalConfig]) -> tuple[Metrics, Raw]:
    cfg = ctx.cfg
    rng = np.random.default_rng(cfg.seed)
    key = jax.random.key(cfg.seed)
    space = make_level_space(
        cfg.level_space.kind,
        ctx.config,
        n_edits=1,
        edit_scale=0.0,
        robot_scale=robot_scale(cfg.robot),
    )
    metrics, raw, key = random_sets(ctx, space, key)
    suite_metrics, suite_raw, key = suite_sets(ctx, rng, key)
    metrics.update(suite_metrics)
    raw.update(suite_raw)
    if isinstance(space, ParamSpace):
        param_metrics, param_raw = param_space_sets(ctx, space, key)
        metrics.update(param_metrics)
        raw.update(param_raw)
    # own streams, so the flat set does not shift the sets above
    flat = space.scaled(
        np.random.default_rng(cfg.seed), cfg.suite_instances, 0.0
    )
    passes = two_passes(ctx, space.render(flat), jax.random.fold_in(key, 1))
    metrics.update(summarize("flat", passes, cfg))
    return metrics, raw


def build(cfg: EvalConfig) -> EvalContext[EvalConfig]:
    config, layout, env = make_eval_env(cfg, cfg.backend)
    make_policy = ppo_networks.make_inference_fn(make_network(env, cfg.network))
    return EvalContext(
        cfg=cfg,
        config=config,
        layout=layout,
        assemble=jax.jit(layout.assemble),
        params=load_params(cfg),
        reset=tile_reset(env),
        unroll=make_unroll(env, make_policy, None),
    )
