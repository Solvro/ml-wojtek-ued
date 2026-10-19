import contextlib
import dataclasses
import functools
from collections.abc import Iterator
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco import mjx

from env.divergence import end_diverged
from env.height_lookup import hfield_geometry
from env.travel import travel_bounds
from env.warp import configure_warp
from robots.terrain import TerrainRobot
from wojtek_env.envs import randomize, registry


class Obs(NamedTuple):
    actor: jax.Array
    critic: jax.Array


class ObsSize(NamedTuple):
    actor: int
    critic: int


def _where_done(done: jax.Array, first: jax.Array, current: jax.Array) -> Any:
    if done.shape and done.shape[0] != first.shape[0]:
        return current
    if done.shape:
        done = jnp.reshape(done, [first.shape[0]] + [1] * (first.ndim - 1))
    return jnp.where(done, first, current)


def reset_done_obs(done: jax.Array, first: Any, current: Any) -> Any:
    return jax.tree.map(functools.partial(_where_done, done), first, current)


def reset_done_data(done: jax.Array, first: Any, current: Any) -> Any:
    def leaf(path: Any, first_leaf: Any, current_leaf: Any) -> Any:
        if any(getattr(p, "name", None) == "_impl" for p in path):
            return current_leaf
        return _where_done(done, first_leaf, current_leaf)

    return jax.tree.map_with_path(leaf, first, current)


class FunctionalEnv:
    """A Wojtek task (or a Playground `robot`) as pure reset / step on the
    shared tiled arena, with per-call `hfield_data` and physics DR."""

    def __init__(
        self,
        task: str = "joystick",
        env_overrides: dict | None = None,
        *,
        episode_length: int | None = None,
        seed: int = 0,
        domain_randomize: bool = False,
        robot: TerrainRobot | None = None,
    ):
        base: Any
        if robot is None:
            base = registry.make_env(task, env_overrides)
            dr_fn = randomize.make_domain_randomize(base.mj_model)
        else:
            base = cast(Any, robot)
            dr_fn = robot.domain_randomize
        length = episode_length or base._config.episode_length
        if getattr(base, "_backend", "jax") == "warp":
            configure_warp()

        self._env = brax_training.EpisodeWrapper(base, length, action_repeat=1)
        # Playground robots read heights straight from hfield_data; Wojtek
        # keeps a lookup
        self._has_lookup = robot is None
        self._hfield_geometry = (
            hfield_geometry(base) if self._has_lookup else None
        )
        self.action_size = base.action_size
        self.travel = travel_bounds(base)
        self._domain_randomize = dr_fn if domain_randomize else None

        probe = self._env.reset(jax.random.key(seed))
        self.observation_size = ObsSize(
            actor=int(probe.obs["state"].shape[-1]),
            critic=int(probe.obs["privileged_state"].shape[-1]),
        )

    def sample_dr(
        self, rng: jax.Array, envs_num: int
    ) -> tuple[mjx.Model, mjx.Model] | tuple[None, None]:
        if self._domain_randomize is None:
            return None, None
        base = cast(Any, self._env.unwrapped)
        dr_rng = jax.random.split(rng, envs_num)
        return self._domain_randomize(base._mjx_model, dr_rng)

    @contextlib.contextmanager
    def _swapped_model(
        self,
        hfield_data: jax.Array | None,
        dr_model: mjx.Model | None,
        tile_xy: jax.Array | None = None,
    ) -> Iterator[None]:
        if hfield_data is None and dr_model is None:
            yield
            return
        base = cast(Any, self._env.unwrapped)
        old_model = base._mjx_model
        old_terrain: Any = base._terrain if self._has_lookup else None
        old_spawn = None if self._has_lookup else base.spawn_xy
        new_model = dr_model if dr_model is not None else old_model
        if hfield_data is not None:
            new_model = new_model.tree_replace({"hfield_data": hfield_data})
            if self._has_lookup:
                base._terrain = self._arena_terrain(
                    old_terrain, hfield_data, tile_xy
                )
            elif tile_xy is not None:
                base.spawn_xy = tile_xy
        base._mjx_model = new_model
        try:
            yield
        finally:
            base._mjx_model = old_model
            if self._has_lookup:
                base._terrain = old_terrain
            else:
                base.spawn_xy = old_spawn

    def _arena_terrain(
        self, terrain: Any, hfield_data: jax.Array, tile_xy: jax.Array | None
    ) -> Any:
        # height scan, base height, foot contact and fall checks read the
        # terrain lookup, not hfield_data
        lookup = self._lookup_from_hfield(hfield_data)
        swapped = dataclasses.replace(terrain, lookup=lookup)
        if tile_xy is None:
            return swapped
        # every spawn table entry points at this env's tile
        origin = jnp.broadcast_to(tile_xy, terrain.origin_xy.shape)
        return dataclasses.replace(swapped, origin_xy=origin)

    def _lookup_from_hfield(self, hfield_data: jax.Array) -> jax.Array:
        if self._hfield_geometry is None:
            raise ValueError(
                "hfield_data swap needs a terrain arena whose height lookup "
                "is the single heightfield's grid (the 'accel' arena); this "
                "env has none. Build it with `just build-accel-arena` and set "
                "terrain.arena=accel"
            )
        return self._hfield_geometry.heights(hfield_data)

    def reset(
        self,
        key: jax.Array,
        hfield_data: jax.Array | None = None,
        dr_model: mjx.Model | None = None,
        tile_xy: jax.Array | None = None,
    ) -> tuple[Obs, Any]:
        """`hfield_data`: the shared arena, one array for every env (vmap
        in_axes None); `tile_xy`: this env's tile centre, where it spawns."""
        if hfield_data is not None and tile_xy is None:
            raise ValueError(
                "reset with an arena hfield_data needs the env's tile_xy "
                "(TileLayout.tile_xy); without it the robot spawns at the "
                "arena centre, between tiles"
            )
        with self._swapped_model(hfield_data, dr_model, tile_xy):
            state: Any = self._env.reset(key)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        state.info["diverged"] = jnp.zeros_like(state.done)
        return self._obs(state), state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        hfield_data: jax.Array | None = None,
        dr_model: mjx.Model | None = None,
    ) -> tuple[Obs, Any, jax.Array, jax.Array, dict]:
        del key
        steps = state.info["steps"]
        state.info["steps"] = jnp.where(
            state.done, jnp.zeros_like(steps), steps
        )
        state = state.replace(done=jnp.zeros_like(state.done))
        with self._swapped_model(hfield_data, dr_model):
            state = cast(Any, self._env.step(state, action))
        state = end_diverged(state)
        done = state.done
        data = reset_done_data(done, state.info["first_data"], state.data)
        obs = reset_done_obs(done, state.info["first_obs"], state.obs)
        state = state.replace(data=data, obs=obs)
        return self._obs(state), state, state.reward, state.done, state.info

    @staticmethod
    def _obs(state: Any) -> Obs:
        return Obs(
            actor=state.obs["state"], critic=state.obs["privileged_state"]
        )
