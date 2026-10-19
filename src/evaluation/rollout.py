import dataclasses
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model as brax_model

from config.hfield_config import HFieldConfig
from config.schema import Backend, EvalConfig, Sim2SimConfig
from env.factory import check_wojtek_arena, make_env
from env.functional import FunctionalEnv
from hfield.tiles import TileLayout
from robots.scale import hfield_config
from training.unroll import Unroll

# per (level, episode): fell, final travel
Passes = list[tuple[np.ndarray, np.ndarray]]


C = TypeVar("C", EvalConfig, Sim2SimConfig)


@dataclasses.dataclass
class EvalContext(Generic[C]):
    cfg: C
    config: HFieldConfig
    layout: TileLayout
    assemble: Callable[[jax.Array], jax.Array]
    params: Any
    reset: Callable[..., Any]
    unroll: Unroll


def levels_per_chunk(cfg: EvalConfig | Sim2SimConfig, layout: TileLayout) -> int:
    """Levels per batch: `chunk_envs` envs, at most one level per tile."""
    return min(max(1, cfg.chunk_envs // cfg.episodes), layout.n_tiles)


def make_eval_env(
    cfg: EvalConfig | Sim2SimConfig, backend: Backend
) -> tuple[HFieldConfig, TileLayout, FunctionalEnv]:
    config = hfield_config(cfg.robot.name, cfg.arena)
    check_wojtek_arena(cfg.robot.name, config)
    layout = TileLayout(config)
    num_envs = levels_per_chunk(cfg, layout) * cfg.episodes
    env = make_env(cfg.robot.name, backend, config, num_envs=num_envs, train=False)
    # scored on the deterministic policy: its own obs and feet, not the critic
    reach = env.travel.reach(cfg.level_steps, env.travel.actor_sensing)
    layout.check_reach(reach, f"level_steps={cfg.level_steps}")
    return config, layout, env


def tile_reset(env: FunctionalEnv) -> Callable[..., Any]:
    """Jitted batched reset: (keys, arena hfield_data, tile_xy per env)."""

    def reset(key: jax.Array, hfield: jax.Array, tile_xy: jax.Array) -> Any:
        return env.reset(key, hfield, tile_xy=tile_xy)

    return jax.jit(jax.vmap(reset, in_axes=(0, None, 0)))


def load_params(cfg: EvalConfig | Sim2SimConfig) -> Any:
    return brax_model.load_params(cfg.checkpoint)


def rollout(
    ctx: EvalContext[Any], hfields: np.ndarray, key: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    """Per (level, episode): fell and final travel. hfields: (levels, N)."""
    cfg, layout = ctx.cfg, ctx.layout
    per_chunk = levels_per_chunk(cfg, layout)
    fell, travel = [], []
    for start in range(0, len(hfields), per_chunk):
        levels = jnp.asarray(hfields[start : start + per_chunk])
        arena = ctx.assemble(levels)
        tile_xy = layout.tile_xy(levels.shape[0], cfg.episodes)
        key, reset_key, act_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, tile_xy.shape[0])
        _, state = ctx.reset(reset_keys, arena, tile_xy)
        start_xy = state.data.qpos[:, :2]
        state, data = ctx.unroll(
            ctx.params,
            state,
            act_key,
            arena,
            None,
            deterministic=True,
            length=cfg.level_steps,
        )
        log = data.extras["log"]
        end_xy = log["xy"][:, -1]
        fell.append(np.asarray(log["fell"].max(axis=1)))
        travel.append(np.asarray(jnp.linalg.norm(end_xy - start_xy, axis=-1)))
    shape = (len(hfields), cfg.episodes)
    return (
        np.concatenate(fell).reshape(shape),
        np.concatenate(travel).reshape(shape),
    )


def two_passes(ctx: EvalContext[Any], hfields: np.ndarray, key: jax.Array) -> Passes:
    """Two independent rollouts of every level (cross-fitted CVaR)."""
    key_a, key_b = jax.random.split(key)
    return [rollout(ctx, hfields, key_a), rollout(ctx, hfields, key_b)]


def level_success(fell: np.ndarray, travel: np.ndarray, threshold: float) -> np.ndarray:
    """Per level: fraction of episodes with no fall and travel >= threshold."""
    return ((fell == 0) & (travel >= threshold)).mean(axis=1)
