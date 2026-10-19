"""Where a training iteration spends its time (docs/ued-accel.md#throughput)."""

import json
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from config.schema import TrainConfig
from training.build import build
from training.context import TrainContext
from training.evaluate import evaluate
from training.iteration import run_levels
from training.ppo import PPO


def timed(fn: Any, repeats: int) -> float:
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / repeats


def env_steps_per_s(ctx: TrainContext, hfield: jax.Array, dr: bool) -> float:
    env, cfg, layout = ctx.env, ctx.cfg, ctx.layout
    n = cfg.num_lanes * cfg.envs_per_lane
    dr_model, dr_axes = env.sample_dr(jax.random.key(1), n) if dr else (None, None)
    tile_xy = layout.tile_xy(cfg.num_lanes, cfg.envs_per_lane)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None, dr_axes, 0)))
    _, state = reset(jax.random.split(jax.random.key(0), n), hfield, dr_model, tile_xy)
    action = jnp.zeros((n, env.action_size))

    def one(s: Any, a: jax.Array, h: jax.Array, m: Any) -> Any:
        return env.step(jax.random.key(0), s, a, h, m)[1]

    v_step = jax.vmap(one, in_axes=(0, 0, None, dr_axes))

    @jax.jit
    def run(state: Any, h: jax.Array, m: Any) -> Any:
        def body(s: Any, _: None) -> tuple[Any, None]:
            return v_step(s, action, h, m), None

        return jax.lax.scan(body, state, None, 20)[0]

    return n * 20 / timed(lambda: run(state, hfield, dr_model), 2)


def iteration_parts(ctx: TrainContext, levels: np.ndarray) -> dict[str, float]:
    env, cfg, student = ctx.env, ctx.cfg, ctx.student
    n = cfg.num_lanes * cfg.envs_per_lane
    rec: dict[str, float] = {}
    start = time.perf_counter()
    hfield = ctx.assemble(jnp.asarray(ctx.space.render(levels)))
    jax.block_until_ready(hfield)
    rec["render_assemble_s"] = time.perf_counter() - start
    rec["sample_dr_s"] = timed(lambda: env.sample_dr(jax.random.key(2), n)[0], 2)
    dr_model, _ = env.sample_dr(jax.random.key(2), n)
    tile_xy = ctx.layout.tile_xy(cfg.num_lanes, cfg.envs_per_lane)
    keys = jax.random.split(jax.random.key(3), n)
    rec["reset_s"] = timed(lambda: ctx.reset(keys, hfield, dr_model, tile_xy), 2)
    _, state = ctx.reset(keys, hfield, dr_model, tile_xy)

    def unroll() -> Any:
        return ctx.unroll(
            student.inference_params(),
            state,
            jax.random.key(4),
            hfield,
            dr_model,
            deterministic=False,
            length=PPO["unroll_length"],
        )

    rec["unroll_s"] = timed(unroll, 3)
    _, data = unroll()
    rec["update_s"] = timed(
        lambda: student.update(
            student.params,
            student.opt_state,
            student.normalizer,
            data,
            jax.random.key(5),
            teacher=None,
            bc_coef=jnp.float32(0.0),
        ),
        3,
    )
    rec["regret_s"] = timed(
        lambda: student.regret(student.params, student.normalizer, data), 3
    )
    return rec


def profile(cfg: TrainConfig) -> None:
    """Times a training iteration's pieces after compilation; one JSON line."""
    start = time.perf_counter()
    ctx = build(cfg)
    rec: dict[str, Any] = {
        "backend": cfg.backend,
        "robot": cfg.robot.name,
        "num_envs": cfg.num_lanes * cfg.envs_per_lane,
        "build_s": time.perf_counter() - start,
    }
    levels = ctx.space.uniform(np.random.default_rng(0), cfg.num_lanes)
    hfield = ctx.assemble(jnp.asarray(ctx.space.render(levels)))
    for dr in (False, True):
        rec[f"env_steps_per_s_dr{int(dr)}"] = env_steps_per_s(ctx, hfield, dr)
    rec.update(iteration_parts(ctx, levels))
    unrolls = cfg.level_steps // PPO["unroll_length"]
    per_unroll = rec["unroll_s"] + rec["update_s"] + rec["regret_s"]
    once = rec["render_assemble_s"] + rec["sample_dr_s"] + rec["reset_s"]
    rec["iteration_est_s"] = once + unrolls * per_unroll
    steps = rec["num_envs"] * cfg.level_steps
    rec["loop_steps_per_s"] = steps / rec["iteration_est_s"]
    run_levels(ctx, levels, train=True)
    start = time.perf_counter()
    run_levels(ctx, levels, train=True)
    rec["run_levels_steps_per_s"] = steps / (time.perf_counter() - start)
    evaluate(ctx, np.random.default_rng(1234))
    start = time.perf_counter()
    evaluate(ctx, np.random.default_rng(1234))
    rec["evaluate_s"] = time.perf_counter() - start
    stats = jax.local_devices()[0].memory_stats() or {}
    rec["peak_gpu_gb"] = stats.get("peak_bytes_in_use", 0) / 1e9
    print(json.dumps(rec), flush=True)
