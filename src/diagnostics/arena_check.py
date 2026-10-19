"""Every env of a batch on its own tile of the shared arena, per backend."""

import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from config.hfield_config import HFieldConfig
from config.schema import Backend, CheckHFieldConfig
from env.functional import FunctionalEnv
from env.overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
)
from hfield.arena import check_arena
from hfield.tiles import TileLayout
from wojtek_env import paths


def make_env(config: HFieldConfig, backend: Backend, num_envs: int) -> FunctionalEnv:
    overrides = merge_overrides(
        accel_arena_overrides(config.pad_radius),
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


def flat_pit_levels(config: HFieldConfig, n: int, pit: float) -> jax.Array:
    """(n, grid**2): even tiles flat, odd tiles a pit `pit` m deep."""
    value = 0.5 - pit / (2 * config.amplitude)
    levels = np.full((n, config.grid_size**2), 0.5, np.float32)
    levels[1::2] = value
    return jnp.asarray(levels)


def settle(
    env: FunctionalEnv,
    layout: TileLayout,
    levels: jax.Array,
    envs_per_tile: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per env: base z per step and dones, zero action."""
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


def bench(cfg: CheckHFieldConfig) -> None:
    env = make_env(cfg.arena, cfg.backend, cfg.bench_envs)
    layout = TileLayout(cfg.arena)
    levels = jax.random.uniform(
        jax.random.key(1), (layout.n_tiles, cfg.arena.grid_size**2)
    )
    per_tile = cfg.bench_envs // layout.n_tiles

    def run() -> None:
        settle(env, layout, levels, per_tile, cfg.steps)

    run()
    start = time.perf_counter()
    run()
    rate = cfg.bench_envs * cfg.steps / (time.perf_counter() - start)
    print(f"bench backend={cfg.backend} envs={cfg.bench_envs}: {rate:,.0f} env-steps/s")


def compare_backends(
    runs: dict[Backend, tuple[np.ndarray, np.ndarray]], cfg: CheckHFieldConfig
) -> bool:
    """Whether jax and warp agree on every tile's terrain; prints the gaps.
    The base-z gap is backend physics, not the arena: warp stands the robot
    ~7-11 mm higher than jax on every tile (job 6109870)."""
    depth = {}
    for backend, (z, _) in runs.items():
        tile_z = late_z(z, cfg.envs_per_tile).mean(axis=1)
        depth[backend] = tile_z[0::2] - tile_z[1::2]
    depth_gap = np.abs(depth[Backend.jax] - depth[Backend.warp]).max()
    z_gap = np.abs(runs[Backend.jax][0] - runs[Backend.warp][0]).max()
    same_dones = np.array_equal(runs[Backend.jax][1], runs[Backend.warp][1])
    ok = depth_gap <= cfg.tol and same_dones
    print(
        f"jax vs warp: pit depth jax {depth[Backend.jax].mean():.4f} / warp "
        f"{depth[Backend.warp].mean():.4f} m, max per-tile gap {depth_gap:.2e} m "
        f"(tol {cfg.tol}), same dones {same_dones}: "
        f"{'PASS' if ok else 'FAIL'}; max base-z gap {z_gap:.2e} m"
    )
    return bool(ok)


def check(cfg: CheckHFieldConfig) -> None:
    """Flat and pit tiles alternate: a pit env must stand `pit` m lower
    than a flat one with no more dones, on its own tile's terrain."""
    check_arena(paths.terrain_paths("accel")["spec"], cfg.arena, "accel")
    if cfg.bench_envs:
        bench(cfg)
        return
    levels = flat_pit_levels(cfg.arena, cfg.tiles, cfg.pit)
    layout = TileLayout(cfg.arena)
    n = cfg.tiles * cfg.envs_per_tile
    backends = [Backend.jax, Backend.warp] if cfg.compare else [cfg.backend]
    runs = {}
    for backend in backends:
        env = make_env(cfg.arena, backend, n)
        runs[backend] = settle(env, layout, levels, cfg.envs_per_tile, cfg.steps)
        report(backend, *runs[backend], cfg.envs_per_tile)
    print(f"expected: pit late z ~{cfg.pit} m below flat, no dones")
    if cfg.compare and not compare_backends(runs, cfg):
        sys.exit(1)
