"""Check that every env of a batch stands on its own tile of the shared arena.

Tiles alternate flat (0.5 everywhere) and pit (the whole patch, pad
included, `--pit` m below the flat plane). Spawn sits at the pad height, so
a pit env drops onto the pit floor and stands `--pit` m lower than a flat
env, with no more dones: physics (the arena hfield) and the terrain lookup
(base height, falls) must both read the env's own tile. Prints the mean
base z over the last quarter and the dones over `--steps` of zero action.

- `--compare`: the same batch on the jax and the warp backend. Fails when
  the backends disagree on a tile's terrain: per-tile pit depth (flat z -
  pit z of the neighbouring tile) differing by more than `--tol` m, or
  different dones. Also prints the largest base-z gap between backends,
  which is backend physics, not the arena: warp stands the robot ~7-11 mm
  higher than jax on every tile (job 6109870).
- `--bench_envs N`: env steps per second on the arena, N envs.

Needs the benchmark `accel` arena (`just build-accel-arena ...`).
"""

import argparse
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from config.hfield_config import HFieldConfig
from hfield.arena import check_arena
from hfield.tiles import TileLayout
from utils.env import FunctionalEnv
from utils.env_overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
)
from wojtek_env import paths

# the benchmark arena brax_ued.py builds
CONFIG = HFieldConfig(amplitude=0.3, coarse_n=13, pad_radius=0.55, pad_taper=0.2)


def make_env(backend: str, num_envs: int) -> FunctionalEnv:
    overrides = merge_overrides(
        accel_arena_overrides(CONFIG.pad_radius),
        movement_overrides(),
        terrain_overrides(),
        # only falls may end an episode, so dones count physical drops
        {
            "sim": {"num_envs": num_envs, "backend": backend},
            "no_progress": {"enable": False},
            "push": {"enable": False},
            "command": {"zero_prob": 1.0, "height": (0.14, 0.14)},
        },
    )
    return FunctionalEnv(task="joystick", env_overrides=overrides)


def flat_pit_levels(n: int, pit: float) -> jax.Array:
    """(n, grid**2): even tiles flat, odd tiles a pit `pit` m deep."""
    value = 0.5 - pit / (2 * CONFIG.amplitude)
    levels = np.full((n, CONFIG.grid_size**2), 0.5, np.float32)
    levels[1::2] = value
    return jnp.asarray(levels)


def settle(
    env: FunctionalEnv, levels: jax.Array, envs_per_tile: int, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per env: base z per step and dones, zero action."""
    layout = TileLayout(CONFIG)
    arena = layout.assemble(levels)
    tile_xy = layout.tile_xy(levels.shape[0], envs_per_tile)
    keys = jax.random.split(jax.random.key(0), tile_xy.shape[0])
    action = jnp.zeros(env.action_size)

    def run(key: jax.Array, tile: jax.Array) -> tuple[jax.Array, jax.Array]:
        _, state = env.reset(key, arena, tile_xy=tile)

        def step(state: Any, _: None) -> tuple[Any, tuple[jax.Array, jax.Array]]:
            _obs, state, _reward, done, _info = env.step(key, state, action, arena)
            return state, (state.data.qpos[2], done)

        _, (z, done) = jax.lax.scan(step, state, None, steps)
        return z, done

    z, done = jax.jit(jax.vmap(run))(keys, tile_xy)
    return np.asarray(z), np.asarray(done)


def late_z(z: np.ndarray, envs_per_tile: int) -> np.ndarray:
    """(tiles, envs_per_tile) mean base z over the last quarter."""
    return z[:, -z.shape[1] // 4 :].mean(axis=1).reshape(-1, envs_per_tile)


def report(backend: str, z: np.ndarray, done: np.ndarray, envs_per_tile: int) -> None:
    late = late_z(z, envs_per_tile)
    dones = done.sum(axis=1).reshape(-1, envs_per_tile)
    print(f"backend={backend} per tile (late z / dones), even=flat odd=pit:")
    for i, (zs, ds) in enumerate(zip(late, dones, strict=True)):
        kind = "pit " if i % 2 else "flat"
        print(f"  tile {i:2d} {kind} z={np.round(zs, 3)} dones={ds}")


def bench(backend: str, num_envs: int, steps: int) -> None:
    env = make_env(backend, num_envs)
    layout = TileLayout(CONFIG)
    levels = jax.random.uniform(
        jax.random.key(1), (layout.n_tiles, CONFIG.grid_size**2)
    )
    per_tile = num_envs // layout.n_tiles
    run = lambda: settle(env, levels, per_tile, steps)
    run()
    start = time.perf_counter()
    run()
    rate = num_envs * steps / (time.perf_counter() - start)
    print(f"bench backend={backend} envs={num_envs}: {rate:,.0f} env-steps/s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="auto", choices=["auto", "jax", "warp"])
    p.add_argument("--pit", type=float, default=0.05, help="m below the flat plane")
    p.add_argument("--tiles", type=int, default=8, help="tiles in use")
    p.add_argument("--envs_per_tile", type=int, default=2)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--compare", action="store_true")
    p.add_argument(
        "--tol",
        type=float,
        default=5e-3,
        help="m, --compare; a misread tile is off by --pit",
    )
    p.add_argument("--bench_envs", type=int, default=0, help="0 = skip timing")
    args = p.parse_args()
    check_arena(paths.terrain_paths("accel")["spec"], CONFIG, "accel")

    if args.bench_envs:
        bench(args.backend, args.bench_envs, args.steps)
        return
    levels = flat_pit_levels(args.tiles, args.pit)
    n = args.tiles * args.envs_per_tile
    backends = ["jax", "warp"] if args.compare else [args.backend]
    runs = {}
    for backend in backends:
        env = make_env(backend, n)
        runs[backend] = settle(env, levels, args.envs_per_tile, args.steps)
        report(backend, *runs[backend], args.envs_per_tile)
    print(f"expected: pit late z ~{args.pit} m below flat, no dones")
    if args.compare:
        depth = {}
        for backend, (z, _) in runs.items():
            tile_z = late_z(z, args.envs_per_tile).mean(axis=1)
            depth[backend] = tile_z[0::2] - tile_z[1::2]
        depth_gap = np.abs(depth["jax"] - depth["warp"]).max()
        z_gap = np.abs(runs["jax"][0] - runs["warp"][0]).max()
        same_dones = np.array_equal(runs["jax"][1], runs["warp"][1])
        ok = depth_gap <= args.tol and same_dones
        print(
            f"jax vs warp: pit depth jax {depth['jax'].mean():.4f} / warp "
            f"{depth['warp'].mean():.4f} m, max per-tile gap {depth_gap:.2e} m "
            f"(tol {args.tol}), same dones {same_dones}: "
            f"{'PASS' if ok else 'FAIL'}; max base-z gap {z_gap:.2e} m"
        )
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
