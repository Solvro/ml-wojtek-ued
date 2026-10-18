"""Final evaluation of a `brax_ued.py` checkpoint, built around CVaR.

Three level sets, deterministic policy, no physics DR, no `no_progress` cut;
an episode succeeds when it never falls and ends at least `min_travel` m
from spawn after `level_steps`:

- `random/d<d>`: `num_levels` fresh levels per difficulty from
  `--level_space` (the training distribution of DR, shrunk toward flat by
  d; the same base levels at every d).
- `flat`: `suite_instances` flat levels (`scaled(0)`) - the no-terrain
  reference.
- `suite/<terrain>_<k>`: a held-out structured suite no method trains on -
  radially symmetric stairs up/down, slopes up/down, trenches, waves and
  random blocks at four strengths, written straight into the arena tiles.

Every level is rolled out in two independent passes (`episodes` each).
CVaR_alpha is cross-fitted: levels ranked worst-alpha on one pass are scored
on the other, both ways, averaged - ranking and scoring on the same
episodes would bias it low by selection noise. 95% CIs bootstrap over
levels. `level_success_corr` (pass A vs pass B per-level success) checks
that levels genuinely differ in difficulty.

With `--level_space param` two more sets:

- `param/uniform`: `num_levels` levels drawn uniformly from the 8-parameter
  space (DR's training distribution) - heterogeneous, so CVaR reads a real
  tail.
- `axis/<feature>`: capability envelopes. One feature ramps from off to its
  full range in `AXIS_STEPS` steps (`suite_instances` seeded levels each),
  everything else off. `envelope` = the largest |feature| (m or deg) still
  passed at 50% success, interpolated; `auc` = mean success over the ramp.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks

from config.hfield_config import HFieldConfig
from hfield.level_space import PARAMS, ParamSpace, make_level_space
from hfield.tiles import TileLayout
from scripts.brax_ued import (
    ROBOT_SCALE,
    add_backend_arg,
    check_wojtek_arena,
    hfield_config,
    make_env,
    make_network,
    make_unroll,
    min_travel,
)

DIFFICULTIES = (0.25, 0.5, 0.75, 1.0)
SUITE_STRENGTHS = (1, 2, 3, 4)


def suite_heights(
    terrain: str, k: int, d: np.ndarray, xy: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Height (m) at strength k; d = radial distance past the spawn pad.

    Radial so that the spawn yaw and command heading do not matter.
    """
    if terrain in ("stairs_up", "stairs_down"):
        h = 0.02 * k * np.ceil(d / 0.3)
        return h if terrain == "stairs_up" else -h
    if terrain in ("slope_up", "slope_down"):
        h = np.tan(np.radians(5.0 * k)) * d
        return h if terrain == "slope_up" else -h
    if terrain == "trenches":
        return np.where((d % 0.5) < 0.05 * k, -0.1, 0.0)
    if terrain == "waves":
        return 0.02 * k * np.sin(2 * np.pi * d / 0.6)
    if terrain == "blocks":
        cells = np.floor(xy / 0.3).astype(int)
        cells -= cells.min(axis=(0, 1))
        heights = rng.uniform(-0.02 * k, 0.02 * k, cells.max(axis=(0, 1)) + 1)
        return heights[cells[..., 0], cells[..., 1]]
    raise ValueError(f"unknown suite terrain {terrain!r}")


SUITE_TERRAINS = (
    "stairs_up",
    "stairs_down",
    "slope_up",
    "slope_down",
    "trenches",
    "waves",
    "blocks",
)


def suite_hfield(
    config: HFieldConfig,
    terrain: str,
    k: int,
    rng: np.random.Generator,
    robot_scale: float,
) -> np.ndarray:
    half = config.patch_radius
    axis = np.linspace(-half, half, config.grid_size)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x**2 + y**2)
    start = config.pad_radius + config.pad_taper
    taper = np.clip((r - config.pad_radius) / config.pad_taper, 0.0, 1.0)
    heights = np.zeros_like(r)
    outside = r >= start
    xy = np.stack([x, y], axis=-1)
    d = np.maximum(r - start, 0.0)
    # geometric scaling: same slopes, every length x robot_scale
    s = robot_scale
    heights[outside] = s * suite_heights(terrain, k, d / s, xy / s, rng)[outside]
    data = 0.5 + heights * taper / (2 * config.amplitude)
    return np.clip(data, 0.0, 1.0).astype(np.float32).reshape(-1)


def rollout(
    ctx: dict[str, Any], hfields: np.ndarray, key: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Per (level, episode): fell and final travel. hfields: (levels, N)."""
    args = ctx["args"]
    layout = ctx["layout"]
    per_chunk = levels_per_chunk(args, layout)
    fell, travel = [], []
    for start in range(0, len(hfields), per_chunk):
        levels = jnp.asarray(hfields[start : start + per_chunk])
        arena = ctx["assemble"](levels)
        tile_xy = layout.tile_xy(levels.shape[0], args.episodes)
        key, reset_key, act_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, tile_xy.shape[0])
        _, state = ctx["reset"](reset_keys, arena, tile_xy)
        start_xy = state.data.qpos[:, :2]
        state, data = ctx["unroll"](
            ctx["params"],
            state,
            act_key,
            arena,
            None,
            deterministic=True,
            length=args.level_steps,
        )
        log = data.extras["log"]
        fell.append(np.asarray(log["fell"].max(axis=1)))
        travel.append(np.asarray(jnp.linalg.norm(log["xy"][:, -1] - start_xy, axis=-1)))
    shape = (len(hfields), args.episodes)
    return np.concatenate(fell).reshape(shape), np.concatenate(travel).reshape(shape)


def levels_per_chunk(args: argparse.Namespace, layout: TileLayout) -> int:
    """Levels per batch: `chunk_envs` envs, at most one level per tile."""
    return min(max(1, args.chunk_envs // args.episodes), layout.n_tiles)


def cross_fit_cvar(a: np.ndarray, b: np.ndarray, alpha: float) -> float:
    """Worst-alpha levels by one pass, scored on the other, both ways."""
    k = max(1, int(np.ceil(alpha * len(a))))
    return 0.5 * (b[np.argsort(a)[:k]].mean() + a[np.argsort(b)[:k]].mean())


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float], n: int, rng: np.random.Generator
) -> tuple[float, float]:
    samples = [statistic(rng.integers(0, n, n)) for _ in range(1000)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(
    prefix: str, passes: list[tuple[np.ndarray, np.ndarray]], args: argparse.Namespace
) -> dict[str, float]:
    (fell_a, travel_a), (fell_b, travel_b) = passes
    success_a = ((fell_a == 0) & (travel_a >= min_travel(args))).mean(axis=1)
    success_b = ((fell_b == 0) & (travel_b >= min_travel(args))).mean(axis=1)
    rng = np.random.default_rng(0)
    n = len(success_a)
    mean = 0.5 * (success_a + success_b)
    out = {
        f"{prefix}/success_rate": float(mean.mean()),
        f"{prefix}/fall_rate": float(0.5 * (fell_a.mean() + fell_b.mean())),
        f"{prefix}/mean_travel": float(0.5 * (travel_a.mean() + travel_b.mean())),
    }
    lo, hi = bootstrap_ci(lambda i: float(mean[i].mean()), n, rng)
    out[f"{prefix}/success_rate_ci_low"], out[f"{prefix}/success_rate_ci_high"] = lo, hi
    for alpha in args.alphas:
        tag = f"{prefix}/cvar{alpha}_success"
        out[tag] = float(cross_fit_cvar(success_a, success_b, alpha))
        lo, hi = bootstrap_ci(
            lambda i, alpha=alpha: cross_fit_cvar(success_a[i], success_b[i], alpha),
            n,
            rng,
        )
        out[f"{tag}_ci_low"], out[f"{tag}_ci_high"] = lo, hi
    if success_a.std() > 0 and success_b.std() > 0:
        out[f"{prefix}/level_success_corr"] = float(
            np.corrcoef(success_a, success_b)[0, 1]
        )
    return out


def two_passes(
    ctx: dict[str, Any], hfields: np.ndarray, key: jax.Array
) -> list[tuple[np.ndarray, np.ndarray]]:
    key_a, key_b = jax.random.split(key)
    return [rollout(ctx, hfields, key_a), rollout(ctx, hfields, key_b)]


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


def envelope(values: np.ndarray, success: np.ndarray, threshold: float = 0.5) -> float:
    """Largest |feature| still passed at `threshold` success, interpolated."""
    if success[0] < threshold:
        return 0.0
    for i in range(1, len(values)):
        if success[i] < threshold:
            frac = (success[i - 1] - threshold) / (success[i - 1] - success[i])
            return float(values[i - 1] + frac * (values[i] - values[i - 1]))
    return float(values[-1])


def param_space_sets(
    ctx: dict[str, Any], space: ParamSpace, key: jax.Array
) -> tuple[dict[str, float], dict[str, Any]]:
    args = ctx["args"]
    rng = np.random.default_rng(args.seed + 1)
    key, sub = jax.random.split(key)
    uniform = space.uniform(rng, args.num_levels)
    passes = two_passes(ctx, space.render(uniform), sub)
    metrics = summarize("param/uniform", passes, args)
    raw = {
        "param_uniform_fell": np.stack([p[0] for p in passes]),
        "param_uniform_travel": np.stack([p[1] for p in passes]),
        "param_uniform_levels": uniform,
    }
    for name, index, end in AXES:
        param = PARAMS[index]
        steps = np.linspace(0.0, 1.0, AXIS_STEPS + 1)[1:]
        physical, success = [], []
        for t in steps:
            genome = np.tile(space.flat, (args.suite_instances, 1))
            genome[:, index] = space.flat[index] + (end - space.flat[index]) * t
            levels = space.from_generated(genome, rng)
            key, sub = jax.random.split(key)
            summary = summarize(
                f"axis/{name}", two_passes(ctx, space.render(levels), sub), args
            )
            physical.append(
                abs(param.low + genome[0, index] * (param.high - param.low))
            )
            success.append(summary[f"axis/{name}/success_rate"])
        physical, success = np.asarray(physical), np.asarray(success)
        for j, (v, sr) in enumerate(zip(physical, success, strict=True)):
            metrics[f"axis/{name}/value{j}"] = float(v)
            metrics[f"axis/{name}/success{j}"] = float(sr)
        metrics[f"axis/{name}/envelope"] = envelope(physical, success)
        metrics[f"axis/{name}/auc"] = float(success.mean())
        print(name, param.name, np.round(physical, 3), np.round(success, 2), flush=True)
    return metrics, raw


def evaluate(ctx: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    args, config = ctx["args"], ctx["config"]
    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    metrics, raw = {}, {}
    space = make_level_space(
        args.level_space,
        config,
        n_edits=1,
        edit_scale=0.0,
        robot_scale=ROBOT_SCALE[args.robot],
    )
    raw["random_levels"] = space.scaled(
        np.random.default_rng(args.seed), args.num_levels, 1.0
    )
    for difficulty in DIFFICULTIES:
        # same seed per difficulty: the same base levels, shrunk toward flat
        levels = space.scaled(
            np.random.default_rng(args.seed), args.num_levels, difficulty
        )
        hfields = space.render(levels)
        key, sub = jax.random.split(key)
        passes = two_passes(ctx, hfields, sub)
        metrics.update(summarize(f"random/d{difficulty}", passes, args))
        raw[f"random_d{difficulty}_fell"] = np.stack([p[0] for p in passes])
        raw[f"random_d{difficulty}_travel"] = np.stack([p[1] for p in passes])
        print(
            {k: round(v, 3) for k, v in metrics.items() if f"d{difficulty}/" in k},
            flush=True,
        )
    suite_success = []
    for terrain in SUITE_TERRAINS:
        for k in SUITE_STRENGTHS:
            hfields = np.stack(
                [
                    suite_hfield(config, terrain, k, rng, ROBOT_SCALE[args.robot])
                    for _ in range(args.suite_instances)
                ]
            )
            key, sub = jax.random.split(key)
            passes = two_passes(ctx, hfields, sub)
            summary = summarize(f"suite/{terrain}_{k}", passes, args)
            metrics[f"suite/{terrain}_{k}/success_rate"] = summary[
                f"suite/{terrain}_{k}/success_rate"
            ]
            suite_success.append(summary[f"suite/{terrain}_{k}/success_rate"])
    suite_success = np.asarray(suite_success)
    metrics["suite/success_rate"] = float(suite_success.mean())
    for alpha in args.alphas:
        k = max(1, int(np.ceil(alpha * len(suite_success))))
        metrics[f"suite/cvar{alpha}_success"] = float(np.sort(suite_success)[:k].mean())
    raw["suite_success"] = suite_success
    if isinstance(space, ParamSpace):
        param_metrics, param_raw = param_space_sets(ctx, space, key)
        metrics.update(param_metrics)
        raw.update(param_raw)
    # own streams, so the sets above keep the levels and keys they had in v8
    flat = space.scaled(np.random.default_rng(args.seed), args.suite_instances, 0.0)
    passes = two_passes(ctx, space.render(flat), jax.random.fold_in(key, 1))
    metrics.update(summarize("flat", passes, args))
    return metrics, raw


def build(args: argparse.Namespace) -> dict[str, Any]:
    config = hfield_config(args)
    check_wojtek_arena(args, config)
    layout = TileLayout(config)
    num_envs = levels_per_chunk(args, layout) * args.episodes
    env = make_env(args, config, num_envs, train=False)
    # scored on the deterministic policy: its own obs and feet, not the critic
    reach = env.travel.reach(args.level_steps, env.travel.actor_sensing)
    layout.check_reach(reach, f"--level_steps {args.level_steps}")
    network = make_network(env)
    make_policy = ppo_networks.make_inference_fn(network)
    return {
        "args": args,
        "config": config,
        "layout": layout,
        "assemble": jax.jit(layout.assemble),
        "params": brax_model.load_params(args.checkpoint),
        "reset": jax.jit(
            jax.vmap(lambda k, h, t: env.reset(k, h, tile_xy=t), in_axes=(0, None, 0))
        ),
        "unroll": make_unroll(env, make_policy, None),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="brax_ued.py .pkl params")
    p.add_argument("--name", required=True, help="wandb run name")
    p.add_argument("--entity", default="")
    p.add_argument("--project", default="wojtyla")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_levels", type=int, default=2048)
    p.add_argument("--episodes", type=int, default=4, help="per level per pass")
    p.add_argument("--suite_instances", type=int, default=64)
    p.add_argument("--chunk_envs", type=int, default=4096)
    p.add_argument("--level_steps", type=int, default=300)
    p.add_argument("--min_travel", type=float, default=1.0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25])
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--level_space", default="grid", choices=["grid", "param"])
    p.add_argument("--robot", default="wojtek", choices=list(ROBOT_SCALE))
    add_backend_arg(p)
    p.add_argument("--coarse_n", type=int, default=13)
    p.add_argument("--pad_radius", type=float, default=0.55)
    p.add_argument("--pad_taper", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    run = wandb.init(
        entity=args.entity or None,
        project=args.project,
        name=args.name,
        tags=["brax-ued", "final-eval", args.robot],
        config=vars(args),
    )
    metrics, raw = evaluate(build(args))
    run.summary.update(metrics)
    run.finish()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / f"{args.name}.npz", **raw)


if __name__ == "__main__":
    main()
