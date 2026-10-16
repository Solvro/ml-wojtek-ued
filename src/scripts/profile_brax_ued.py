"""Where does a `brax_ued.py` iteration spend its time? Prints one JSON line.

Builds the run like `brax_ued.py --mode dr` (param levels) on `--backend`
and times, after compilation: env steps alone (zero action, with and
without physics DR), then one iteration's pieces - level render + arena
assembly, DR draw, reset, a 20-step policy unroll, the PPO update, the
regret pass - and one `evaluate` call. `loop_steps_per_s` is the iteration
estimate from those pieces; `run_levels_steps_per_s` times a real
`run_levels` iteration. Peak GPU memory from jax's allocator.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from scripts import brax_ued


def timed(fn: Any, repeats: int) -> float:
    jax.block_until_ready(fn())
    start = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return (time.perf_counter() - start) / repeats


def env_steps_per_s(ctx: dict[str, Any], hfield: jax.Array, dr: bool) -> float:
    env, args, layout = ctx["env"], ctx["args"], ctx["layout"]
    n = args.num_lanes * args.envs_per_lane
    dr_model, dr_axes = env.sample_dr(jax.random.key(1), n) if dr else (None, None)
    tile_xy = layout.tile_xy(args.num_lanes, args.envs_per_lane)
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


def iteration_parts(ctx: dict[str, Any], levels: np.ndarray) -> dict[str, float]:
    env, args, student = ctx["env"], ctx["args"], ctx["student"]
    n = args.num_lanes * args.envs_per_lane
    rec: dict[str, float] = {}
    start = time.perf_counter()
    hfield = ctx["assemble"](jnp.asarray(ctx["space"].render(levels)))
    jax.block_until_ready(hfield)
    rec["render_assemble_s"] = time.perf_counter() - start
    rec["sample_dr_s"] = timed(lambda: env.sample_dr(jax.random.key(2), n)[0], 2)
    dr_model, _ = env.sample_dr(jax.random.key(2), n)
    tile_xy = ctx["layout"].tile_xy(args.num_lanes, args.envs_per_lane)
    keys = jax.random.split(jax.random.key(3), n)
    rec["reset_s"] = timed(lambda: ctx["reset"](keys, hfield, dr_model, tile_xy), 2)
    _, state = ctx["reset"](keys, hfield, dr_model, tile_xy)

    def unroll() -> Any:
        return ctx["unroll"](
            student.inference_params(),
            state,
            jax.random.key(4),
            hfield,
            dr_model,
            deterministic=False,
            length=brax_ued.PPO["unroll_length"],
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
        ),
        3,
    )
    rec["regret_s"] = timed(
        lambda: student.regret(student.params, student.normalizer, data), 3
    )
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=["jax", "warp"])
    p.add_argument("--profile", default="benchmark", choices=list(brax_ued.ARENAS))
    p.add_argument("--num_lanes", type=int, default=256)
    p.add_argument("--envs_per_lane", type=int, default=32)
    p.add_argument("--level_steps", type=int, default=300)
    p.add_argument("--eval_levels", type=int, default=512)
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        backend = Path(tmp) / "backend.json"
        backend.write_text(json.dumps({"sim": {"backend": a.backend}}))
        args = brax_ued.parse_args(
            [
                *("--mode", "dr", "--name", "profile", "--profile", a.profile),
                *("--level_space", "param", "--n_edits", "2", "--edit_scale", "0.1"),
                *("--num_lanes", str(a.num_lanes)),
                *("--envs_per_lane", str(a.envs_per_lane)),
                *("--level_steps", str(a.level_steps)),
                *("--eval_levels", str(a.eval_levels)),
                *("--env_override", str(backend)),
            ]
        )
        start = time.perf_counter()
        ctx = brax_ued.build(args)
    rec: dict[str, Any] = {
        "backend": a.backend,
        "profile": a.profile,
        "num_envs": a.num_lanes * a.envs_per_lane,
        "build_s": time.perf_counter() - start,
    }
    levels = ctx["space"].uniform(np.random.default_rng(0), args.num_lanes)
    hfield = ctx["assemble"](jnp.asarray(ctx["space"].render(levels)))
    for dr in (False, True):
        rec[f"env_steps_per_s_dr{int(dr)}"] = env_steps_per_s(ctx, hfield, dr)
    rec.update(iteration_parts(ctx, levels))
    unrolls = args.level_steps // brax_ued.PPO["unroll_length"]
    per_unroll = rec["unroll_s"] + rec["update_s"] + rec["regret_s"]
    once = rec["render_assemble_s"] + rec["sample_dr_s"] + rec["reset_s"]
    rec["iteration_est_s"] = once + unrolls * per_unroll
    steps = rec["num_envs"] * args.level_steps
    rec["loop_steps_per_s"] = steps / rec["iteration_est_s"]
    brax_ued.run_levels(ctx, levels, train=True)
    start = time.perf_counter()
    brax_ued.run_levels(ctx, levels, train=True)
    rec["run_levels_steps_per_s"] = steps / (time.perf_counter() - start)
    brax_ued.evaluate(ctx, np.random.default_rng(1234))
    start = time.perf_counter()
    brax_ued.evaluate(ctx, np.random.default_rng(1234))
    rec["evaluate_s"] = time.perf_counter() - start
    stats = jax.local_devices()[0].memory_stats() or {}
    rec["peak_gpu_gb"] = stats.get("peak_bytes_in_use", 0) / 1e9
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
