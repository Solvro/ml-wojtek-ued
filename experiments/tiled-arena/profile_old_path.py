"""Where did brax_ued's time go before the tiled arena? Runs on 4b91bce only.

One variant per process, prints JSON lines.

Variants:
- `cur_jax`: today's loop, per-env `hfield_data` + DR model swap, jax backend.
- `shared_jax` / `shared_warp`: one level baked into the model for every env
  (no per-env hfield swap), jax or warp backend. This is the physics a shared
  tiled arena would run.
"""

import argparse
import dataclasses
import json
import time
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


def bake_level(env: Any, hfield: np.ndarray) -> None:
    base = env._env.unwrapped
    data = jnp.asarray(hfield)
    base._mjx_model = base._mjx_model.tree_replace({"hfield_data": data})
    if env._hfield_geometry is not None:
        lookup = env._hfield_geometry.heights(data)
        base._terrain = dataclasses.replace(base._terrain, lookup=lookup)


def env_only(ctx: dict[str, Any], hfield: Any, dr: bool, steps: int) -> float:
    env, args = ctx["env"], ctx["args"]
    n = args.num_lanes * args.envs_per_lane
    dr_model, dr_axes = env.sample_dr(jax.random.key(1), n) if dr else (None, None)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, 0, dr_axes)))
    _, state = reset(jax.random.split(jax.random.key(0), n), hfield, dr_model)
    action = jnp.zeros((n, env.action_size))

    def one(s: Any, a: jax.Array, h: Any, m: Any) -> Any:
        return env.step(jax.random.key(0), s, a, h, m)[1]

    v_step = jax.vmap(one, in_axes=(0, 0, 0, dr_axes))

    @jax.jit
    def run(state: Any, h: Any, m: Any) -> Any:
        return jax.lax.scan(
            lambda s, _: (v_step(s, action, h, m), None), state, None, steps
        )[0]

    return n * steps / timed(lambda: run(state, hfield, dr_model), 2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variant", required=True, choices=["cur_jax", "shared_jax", "shared_warp"]
    )
    p.add_argument("--profile", default="benchmark")
    p.add_argument("--level_steps", type=int, default=100)
    p.add_argument("--num_lanes", type=int, default=256)
    p.add_argument("--envs_per_lane", type=int, default=32)
    p.add_argument("--eval_levels", type=int, default=512)
    a = p.parse_args()
    argv = [
        "--mode",
        "dr",
        "--name",
        "profile",
        "--profile",
        a.profile,
        "--level_space",
        "param",
        "--n_edits",
        "2",
        "--edit_scale",
        "0.1",
        "--level_steps",
        str(a.level_steps),
        "--num_lanes",
        str(a.num_lanes),
        "--envs_per_lane",
        str(a.envs_per_lane),
        "--eval_levels",
        str(a.eval_levels),
    ]
    args = brax_ued.parse_args(argv)
    backend = "warp" if a.variant == "shared_warp" else "jax"
    original = brax_ued.env_overrides

    def overrides(args: argparse.Namespace) -> dict[str, Any] | None:
        out = original(args)
        assert out is not None
        sim = {"backend": backend, "naconmax_per_env": 116, "njmax": 1024}
        return brax_ued.merge_overrides(out, {"sim": sim})

    # patched in place: 4b91bce's brax_ued has no backend or env hook
    brax_ued.env_overrides = overrides  # ty: ignore[invalid-assignment]
    made: list[Any] = []
    make_env = brax_ued.make_env

    def recording_make_env(*margs: Any, **kw: Any) -> Any:
        made.append(make_env(*margs, **kw))
        return made[-1]

    brax_ued.make_env = recording_make_env  # ty: ignore[invalid-assignment]
    t0 = time.perf_counter()
    ctx = brax_ued.build(args)
    rec: dict[str, Any] = {
        "variant": a.variant,
        "profile": a.profile,
        "build_s": time.perf_counter() - t0,
    }
    space, rng = ctx["space"], np.random.default_rng(0)
    n = args.num_lanes * args.envs_per_lane

    t0 = time.perf_counter()
    levels = space.uniform(rng, args.num_lanes)
    rendered = space.render(levels)
    rec["render_s"] = time.perf_counter() - t0
    if a.variant == "cur_jax":
        hfield = jnp.repeat(jnp.asarray(rendered), args.envs_per_lane, axis=0)
        eval_hfield = jnp.repeat(
            jnp.asarray(rendered[:1]), 2 * args.eval_levels, axis=0
        )
    else:
        bake_level(ctx["env"], rendered[0])
        bake_level(made[1], rendered[0])
        hfield = eval_hfield = None

    for dr in (False, True):
        t0 = time.perf_counter()
        rate = env_only(ctx, hfield, dr, 20)
        rec[f"env_only_dr{int(dr)}_steps_per_s"] = rate
        rec[f"env_only_dr{int(dr)}_total_s"] = time.perf_counter() - t0
        print(json.dumps(rec), flush=True)

    env, student = ctx["env"], ctx["student"]
    t0 = time.perf_counter()
    dr_model, _ = env.sample_dr(jax.random.key(2), n)
    jax.block_until_ready(dr_model)
    rec["sample_dr_s"] = time.perf_counter() - t0
    keys = jax.random.split(jax.random.key(3), n)
    rec["reset_s"] = timed(lambda: ctx["reset"](keys, hfield, dr_model), 2)
    _, state = ctx["reset"](keys, hfield, dr_model)

    def unroll() -> Any:
        return ctx["unroll"](
            student.inference_params(),
            state,
            jax.random.key(4),
            hfield,
            dr_model,
            deterministic=False,
            length=20,
        )

    rec["unroll20_s"] = timed(unroll, 3)
    _, data = unroll()

    def update() -> Any:
        return student.update(
            student.params,
            student.opt_state,
            student.normalizer,
            data,
            jax.random.key(5),
        )

    rec["update_s"] = timed(update, 3)
    rec["regret_s"] = timed(
        lambda: student.regret(student.params, student.normalizer, data), 3
    )
    unrolls = args.level_steps // 20
    iteration = (
        rec["sample_dr_s"]
        + rec["reset_s"]
        + rec["render_s"]
        + unrolls * (rec["unroll20_s"] + rec["update_s"] + rec["regret_s"])
    )
    rec["iteration_est_s"] = iteration
    rec["loop_steps_per_s_no_eval"] = n * args.level_steps / iteration
    rec["unroll_steps_per_s"] = n * 20 / rec["unroll20_s"]

    def evaluate_one() -> Any:
        ekeys = jax.random.split(jax.random.key(6), 2 * args.eval_levels)
        _, s = ctx["eval_reset"](ekeys, eval_hfield)
        return ctx["eval_unroll"](
            student.inference_params(),
            s,
            jax.random.key(7),
            eval_hfield,
            None,
            deterministic=True,
            length=args.level_steps,
        )

    one_difficulty = timed(evaluate_one, 1)
    rec["eval_s_per_call"] = len(brax_ued.EVAL_DIFFICULTIES) * one_difficulty
    stats = jax.local_devices()[0].memory_stats() or {}
    rec["peak_gpu_gb"] = stats.get("peak_bytes_in_use", 0) / 1e9

    t0 = time.perf_counter()
    out = (
        brax_ued.run_levels(ctx, levels, train=True) if a.variant == "cur_jax" else None
    )
    rec["run_levels_s"] = time.perf_counter() - t0 if out is not None else None
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
