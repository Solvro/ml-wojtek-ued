import contextlib
import dataclasses
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from config.hfield_config import HFieldConfig
from modules.ppo_modules import Obs
from robots.go1 import Go1Terrain
from wojtek_env.envs import randomize, registry


class ObsSize(NamedTuple):
    actor: int
    critic: int


class HFieldGeometry(NamedTuple):
    nrow: int
    ncol: int
    base_z: float
    elevation: float

    def heights(self, hfield_data: jax.Array) -> jax.Array:
        grid = hfield_data.reshape(self.nrow, self.ncol)
        return self.base_z + grid * self.elevation


def hfield_geometry(base: Any) -> HFieldGeometry | None:
    if base._terrain is None:
        return None
    m = base.mj_model
    # mujoco is a C extension that ty cannot introspect
    hfield_type = mujoco.mjtGeom.mjGEOM_HFIELD  # ty: ignore[unresolved-attribute]
    geoms = [i for i in range(m.ngeom) if m.geom_type[i] == hfield_type]
    if m.nhfield != 1 or len(geoms) != 1:
        return None
    geometry = HFieldGeometry(
        nrow=int(m.hfield_nrow[0]),
        ncol=int(m.hfield_ncol[0]),
        base_z=float(m.geom_pos[geoms[0], 2]),
        elevation=float(m.hfield_size[0, 2]),
    )
    if base._terrain.lookup.shape != (geometry.nrow, geometry.ncol):
        return None
    return geometry


class FunctionalEnv:
    def __init__(
        self,
        task: str = "joystick",
        env_overrides: dict | None = None,
        episode_length: int | None = None,
        seed: int = 0,
        domain_randomize: bool = False,
        go1_hfield: HFieldConfig | None = None,
        dr_config: dict | None = None,
    ):
        # go1_hfield set: Playground's Go1 on that arena instead of Wojtek
        base: Any
        if go1_hfield is None:
            base = registry.make_env(task, env_overrides)
            dr_fn = randomize.make_domain_randomize(base.mj_model, dr_config)
        else:
            base = cast(Any, Go1Terrain(go1_hfield))
            dr_fn = go1_randomize.domain_randomize
        length = episode_length or base._config.episode_length

        self._env = brax_training.EpisodeWrapper(base, length, action_repeat=1)
        # Go1 reads heights straight from hfield_data; Wojtek keeps a lookup
        self._has_lookup = go1_hfield is None
        self._hfield_geometry = hfield_geometry(base) if self._has_lookup else None
        self.action_size = base.action_size
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
    def _swapped_model(self, hfield_data: jax.Array | None, dr_model: mjx.Model | None):
        if hfield_data is None and dr_model is None:
            yield
            return
        base = cast(Any, self._env.unwrapped)
        old_model = base._mjx_model
        old_terrain: Any = base._terrain if self._has_lookup else None
        new_model = dr_model if dr_model is not None else old_model
        if hfield_data is not None:
            new_model = new_model.tree_replace({"hfield_data": hfield_data})
            if self._has_lookup:
                base._terrain = dataclasses.replace(
                    old_terrain, lookup=self._lookup_from_hfield(hfield_data)
                )
        base._mjx_model = new_model
        try:
            yield
        finally:
            base._mjx_model = old_model
            if self._has_lookup:
                base._terrain = old_terrain

    def _lookup_from_hfield(self, hfield_data: jax.Array) -> jax.Array:
        # height scan, base height, foot contact and fall checks read the
        # terrain lookup, not hfield_data; keep it on the swapped level
        if cast(Any, self._env.unwrapped)._backend == "warp":
            raise ValueError(
                "per-env hfield_data on the warp backend collides every env "
                "with the first env's level (check_hfield_batch.py). Set "
                "sim.backend=jax, e.g. via utils.env_overrides."
                "accel_arena_overrides"
            )
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
    ) -> tuple[Obs, Any]:
        with self._swapped_model(hfield_data, dr_model):
            state: Any = self._env.reset(key)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
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
        state.info["steps"] = jnp.where(state.done, jnp.zeros_like(steps), steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        with self._swapped_model(hfield_data, dr_model):
            state = cast(Any, self._env.step(state, action))

        done = state.done

        def where_done(first, cur):
            d = done
            if d.shape and d.shape[0] != first.shape[0]:
                return cur
            if d.shape:
                d = jnp.reshape(d, [first.shape[0]] + [1] * (len(first.shape) - 1))
            return jnp.where(d, first, cur)

        def reset_leaf(path, first, cur):
            if any(getattr(p, "name", None) == "_impl" for p in path):
                return cur
            return where_done(first, cur)

        data = jax.tree.map_with_path(reset_leaf, state.info["first_data"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        state = state.replace(data=data, obs=obs)
        return self._obs(state), state, state.reward, state.done, state.info

    @staticmethod
    def _obs(state: Any) -> Obs:
        return Obs(actor=state.obs["state"], critic=state.obs["privileged_state"])
