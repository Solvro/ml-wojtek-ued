from collections.abc import Callable
from typing import Any

import jax

from config.hfield_config import HFieldConfig
from config.schema import Backend, Robot
from env.functional import FunctionalEnv
from env.overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
    terrain_task_overrides,
)
from hfield.arena import check_arena
from robots.registry import TERRAIN_ROBOTS
from wojtek_env import paths


def check_wojtek_arena(robot: Robot, config: HFieldConfig) -> None:
    # Playground robots build their arena from `config` at construction;
    # Wojtek loads it
    if robot == Robot.wojtek:
        check_arena(paths.terrain_paths("accel")["spec"], config, "accel")


def make_env(
    robot: Robot,
    backend: Backend,
    config: HFieldConfig,
    *,
    num_envs: int,
    train: bool,
) -> FunctionalEnv:
    if robot in TERRAIN_ROBOTS:
        terrain_robot = TERRAIN_ROBOTS[robot](config, backend=backend)
        return FunctionalEnv(domain_randomize=train, robot=terrain_robot)
    overrides = wojtek_overrides(config, backend, num_envs, train)
    return FunctionalEnv(
        task="joystick", env_overrides=overrides, domain_randomize=train
    )


def wojtek_overrides(
    config: HFieldConfig, backend: Backend, num_envs: int, train: bool
) -> dict[str, Any]:
    return merge_overrides(
        accel_arena_overrides(config.pad_radius, backend),
        movement_overrides(),
        terrain_overrides(),
        terrain_task_overrides(),
        {"sim": {"num_envs": num_envs}},
        # eval scores falls and travel; the stochastic cut only adds noise
        {} if train else {"no_progress": {"enable": False}},
    )


def tile_reset(env: FunctionalEnv) -> Callable[..., Any]:
    """Jitted batched reset: (keys, arena hfield_data, tile_xy per env)."""

    def reset(key: jax.Array, hfield: jax.Array, tile_xy: jax.Array) -> Any:
        return env.reset(key, hfield, tile_xy=tile_xy)

    return jax.jit(jax.vmap(reset, in_axes=(0, None, 0)))
