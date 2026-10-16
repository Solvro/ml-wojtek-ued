import contextlib
import dataclasses
import math
import os
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import mujoco
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from config.hfield_config import HFieldConfig
from modules.ppo_modules import Obs
from robots.go1 import GO1_SCALE, SCAN_X, SCAN_Y, Go1Terrain
from wojtek_env.envs import randomize, registry
from wojtek_env.terrain.terrain_env import SPAWN_EDGE_MARGIN


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


class TravelBounds(NamedTuple):
    """How far from its tile centre an env can read terrain after a reset."""

    jitter: float  # m, spawn offset from the tile centre (Chebyshev)
    speed: float  # m/s, fastest planar command
    # m from the base: feet, or the height scan if any obs reads it (critic)
    sensing: float
    # the same for a deterministic policy rollout: feet and the actor's obs
    actor_sensing: float
    dt: float  # s per env step

    def reach(self, steps: int, sensing: float) -> float:
        return self.jitter + steps * self.dt * self.speed + sensing


def max_command_speed(command: Any) -> float:
    """Fastest planar speed (m/s) Wojtek's command sampler can draw."""
    box = math.hypot(max(map(abs, command.vx)), max(map(abs, command.vy)))
    bias = command.terrain_bias
    # each mode redraws vx from its own range and zeroes vy
    modes = {
        "arc_prob": "arc_vx",
        "pure_slow_prob": "slow_vx",
        "pure_fast_prob": "fast_vx",
        "pure_back_prob": "back_vx",
    }
    straight = [
        max(map(abs, command[vx]))
        for prob, vx in modes.items()
        if command.get(prob, 0.0) > 0 or (bias.enable and bias.get(prob, 0.0) > 0)
    ]
    return max([box, *straight])


def scan_reach(x_range: Any, y_range: Any) -> float:
    return math.hypot(max(map(abs, x_range)), max(map(abs, y_range)))


def travel_bounds(base: Any) -> TravelBounds:
    if isinstance(base, Go1Terrain):
        low, high = base._cmd_low, base._cmd_high
        speed = math.hypot(
            float(max(abs(low[0]), abs(high[0]))),
            float(max(abs(low[1]), abs(high[1]))),
        )
        feet = GO1_SCALE * SPAWN_EDGE_MARGIN
        # Go1's scan is critic-only
        sensing = max(feet, GO1_SCALE * scan_reach(SCAN_X, SCAN_Y))
        return TravelBounds(0.0, speed, sensing, feet, base.dt)
    c = base._config
    hs = c.height_scan
    scan = scan_reach(hs.x_range, hs.y_range) if hs.enable else 0.0
    actor_scan = any(k.startswith("height_scan") for k in c.obs.state)
    return TravelBounds(
        jitter=float(c.terrain.pad_jitter),
        speed=max_command_speed(c.command),
        sensing=max(SPAWN_EDGE_MARGIN, scan),
        actor_sensing=max(SPAWN_EDGE_MARGIN, scan if actor_scan else 0.0),
        dt=base.dt,
    )


# CUDA graphs mjx's warp FFI keeps per callable (its default: 32)
WARP_GRAPH_CACHE = 4


def configure_warp() -> None:
    require_warp_memory()
    # mjx's warp FFI captures one CUDA graph per set of buffer addresses, each
    # owning its own collision scratch (GBs at 8192 envs). Upstream's single
    # jitted ppo.train step reuses its buffers; our separate jitted reset /
    # unroll / eval calls pass new addresses every call, and the default cache
    # ran the GPU out of memory by the third iteration (job 6110159).
    # mjx's vendored warp; the setting has no public import path
    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi

    ffi.set_jax_callable_default_graph_cache_max(WARP_GRAPH_CACHE)


def require_warp_memory() -> None:
    # warp allocates its collision scratch outside JAX's memory pool on every
    # step; with JAX's default 75% preallocation it runs out mid-run
    # (upstream w01-tek: mandatory on terrain)
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError(
            "sim.backend=warp needs XLA_PYTHON_CLIENT_PREALLOCATE=false in the "
            "environment before the process starts (the justfile and the "
            "slurm/ scripts export it); otherwise warp runs out of GPU memory "
            "mid-run. Or run on sim.backend=jax"
        )


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
        if getattr(base, "_backend", "jax") == "warp":
            configure_warp()

        self._env = brax_training.EpisodeWrapper(base, length, action_repeat=1)
        # Go1 reads heights straight from hfield_data; Wojtek keeps a lookup
        self._has_lookup = go1_hfield is None
        self._hfield_geometry = hfield_geometry(base) if self._has_lookup else None
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
    ):
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
                terrain = dataclasses.replace(
                    old_terrain, lookup=self._lookup_from_hfield(hfield_data)
                )
                if tile_xy is not None:
                    # every spawn table entry points at this env's tile
                    origin = jnp.broadcast_to(tile_xy, old_terrain.origin_xy.shape)
                    terrain = dataclasses.replace(terrain, origin_xy=origin)
                base._terrain = terrain
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

    def _lookup_from_hfield(self, hfield_data: jax.Array) -> jax.Array:
        # height scan, base height, foot contact and fall checks read the
        # terrain lookup, not hfield_data; keep it on the arena's levels
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
        """`hfield_data`: the shared tiled arena (`hfield.tiles`), the same
        array for every env (vmap it with in_axes None); `tile_xy`: this
        env's tile centre, where it spawns."""
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
