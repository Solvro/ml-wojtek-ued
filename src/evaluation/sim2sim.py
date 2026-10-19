import dataclasses
import time
from typing import Any, cast

import jax
import numpy as np
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks

from config.hfield_config import HFieldConfig
from config.schema import Backend, LevelSpaceKind, Sim2SimConfig
from env.factory import tile_reset
from evaluation.cpu_physics import CpuPhysics, cpu_physics, worker_count
from evaluation.physics_variants import VARIANTS
from evaluation.rollout import (
    EvalContext,
    level_success,
    load_params,
    make_eval_env,
    rollout,
)
from evaluation.stats import bootstrap_ci
from evaluation.suite import SUITE_STRENGTHS, suite_levels
from hfield.level_space import make_level_space
from robots.scale import ROBOT_SCALE, min_travel
from training.ppo import make_network
from training.unroll import Unroll, make_unroll

# eval_brax's default num_levels; param/uniform is a prefix of its draw
EVAL_BRAX_UNIFORM = 2048
SIM2SIM_STRENGTHS = SUITE_STRENGTHS[:2]

# variant -> set -> (fell, travel, diverged env steps)
SetResult = tuple[np.ndarray, np.ndarray, int]
Results = dict[str, dict[str, SetResult]]


@dataclasses.dataclass
class Sim2SimContext(EvalContext[Sim2SimConfig]):
    base: Any
    physics: CpuPhysics


def level_sets(cfg: Sim2SimConfig, config: HFieldConfig) -> dict[str, Any]:
    """Rendered hfields per set, the levels eval_brax builds for them."""
    scale = ROBOT_SCALE[cfg.robot.name]
    space = make_level_space(
        LevelSpaceKind.param,
        config,
        n_edits=1,
        edit_scale=0.0,
        robot_scale=scale,
    )
    flat = space.scaled(
        np.random.default_rng(cfg.seed), cfg.suite_instances, 0.0
    )
    uniform = space.uniform(
        np.random.default_rng(cfg.seed + 1), EVAL_BRAX_UNIFORM
    )
    sets = {
        "flat": space.render(flat),
        "param/uniform": space.render(uniform[: cfg.num_levels]),
    }
    rng = np.random.default_rng(cfg.seed)
    sets.update(
        suite_levels(config, SIM2SIM_STRENGTHS, cfg.suite_instances, rng, scale)
    )
    return sets


def with_ci(
    out: dict[str, float],
    tag: str,
    per_level: np.ndarray,
    rng: np.random.Generator,
) -> None:
    out[tag] = float(per_level.mean())
    n = len(per_level)
    lo, hi = bootstrap_ci(lambda i: float(per_level[i].mean()), n, rng)
    out[f"{tag}_ci_low"], out[f"{tag}_ci_high"] = lo, hi


def set_metrics(
    prefix: str,
    fell: np.ndarray,
    travel: np.ndarray,
    threshold: float,
    reference: np.ndarray | None,
) -> dict[str, float]:
    """fell / travel: (levels, episodes). reference: mjx_match success."""
    rng = np.random.default_rng(0)
    success = level_success(fell, travel, threshold)
    out: dict[str, float] = {}
    with_ci(out, f"{prefix}/success_rate", success, rng)
    with_ci(out, f"{prefix}/fall_rate", fell.mean(axis=1), rng)
    with_ci(out, f"{prefix}/mean_travel", travel.mean(axis=1), rng)
    if reference is not None:
        with_ci(out, f"{prefix}/drop", reference - success, rng)
    return out


def pool_suite(sets: dict[str, SetResult]) -> dict[str, SetResult]:
    pooled = {k: v for k, v in sets.items() if not k.startswith("suite/")}
    cells = [v for k, v in sets.items() if k.startswith("suite/")]
    pooled["suite"] = (
        np.concatenate([c[0] for c in cells]),
        np.concatenate([c[1] for c in cells]),
        sum(c[2] for c in cells),
    )
    return pooled


def summarize(results: Results, threshold: float) -> dict[str, float]:
    """Per variant x set metrics; `drop` against `mjx_match` on each set."""
    metrics: dict[str, float] = {}
    reference: dict[str, np.ndarray] = {}
    for variant, sets in results.items():
        is_reference = variant == "mjx_match"
        for name, (fell, travel, diverged) in pool_suite(sets).items():
            prefix = f"{variant}/{name}"
            if is_reference:
                reference[name] = level_success(fell, travel, threshold)
            ref = None if is_reference else reference[name]
            metrics.update(set_metrics(prefix, fell, travel, threshold, ref))
            metrics[f"{prefix}/diverged_per_episode"] = diverged / fell.size
        for name, (fell, travel, _) in sets.items():
            if name.startswith("suite/"):
                success = ((fell == 0) & (travel >= threshold)).mean()
                metrics[f"{variant}/{name}/success_rate"] = float(success)
    return metrics


def evaluate(
    ctx: Sim2SimContext, sets: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Every physics variant on every set, with the same reset keys."""
    cfg, base, physics = ctx.cfg, ctx.base, ctx.physics
    results: Results = {}
    raw: dict[str, Any] = {}
    elapsed: dict[str, float] = {}
    for variant, (tests, transform) in VARIANTS.items():
        print(f"{variant}: {tests}", flush=True)
        start = time.time()
        physics.use(transform(base.mj_model, base))
        results[variant] = {}
        for i, (name, hfields) in enumerate(sets.items()):
            # same key per set across variants: paired comparisons
            key = jax.random.fold_in(jax.random.key(cfg.seed), i)
            fell, travel = rollout(ctx, hfields, key)
            results[variant][name] = (fell, travel, physics.take_diverged())
            tag = f"{variant}_{name.replace('/', '_')}"
            raw[f"{tag}_fell"], raw[f"{tag}_travel"] = fell, travel
        elapsed[variant] = time.time() - start
        print(f"{variant}: {elapsed[variant]:.0f} s", flush=True)
    threshold = min_travel(cfg.robot.name, cfg.min_travel)
    metrics = summarize(results, threshold)
    for variant, seconds in elapsed.items():
        metrics[f"{variant}/elapsed_s"] = seconds
    return metrics, raw


def cpu_unroll(physics: CpuPhysics, unroll: Unroll) -> Unroll:
    """`unroll` with the physics stepped by `physics` on CPU MuJoCo."""

    def run(
        inference: Any,
        state: Any,
        key: jax.Array,
        hfield: jax.Array,
        dr_model: Any,
        *,
        deterministic: bool,
        length: int,
    ) -> tuple[Any, types.Transition]:
        physics.set_arena(np.asarray(hfield))
        with cpu_physics(physics):
            return unroll(
                inference,
                state,
                key,
                hfield,
                dr_model,
                deterministic=deterministic,
                length=length,
            )

    return run


def build(cfg: Sim2SimConfig) -> Sim2SimContext:
    # CpuPhysics steps the jax env's data layout on CPU MuJoCo
    config, layout, env = make_eval_env(cfg, Backend.jax)
    base = cast(Any, env)._env.unwrapped
    physics = CpuPhysics(base.mj_model, worker_count())
    make_policy = ppo_networks.make_inference_fn(make_network(env))
    unroll = make_unroll(env, make_policy, None)
    return Sim2SimContext(
        cfg=cfg,
        config=config,
        layout=layout,
        assemble=jax.jit(layout.assemble),
        params=load_params(cfg),
        reset=tile_reset(env),
        unroll=cpu_unroll(physics, unroll),
        base=base,
        physics=physics,
    )
