"""Shared machinery for Playground quadrupeds on the UED heightfield arena.

A robot file (`go1.py`, ...) keeps its constants and what differs; this base
gives every Playground robot the same benchmark as Wojtek:

- the scene is the tiled ACCEL arena (`hfield.tiles`): one swappable
  `hfield` geom (`floor`, geom 0 so a friction DR keyed on geom 0 still hits
  it) sized by `HFieldConfig`; the robot spawns on `spawn_xy`, its tile
  centre, which `FunctionalEnv` sets around reset;
- commands walk forward, `SCALE` times Wojtek's range, so the robot leaves
  the spawn pad and meets the level within an episode;
- a fall is Wojtek's rule at the robot's size: trunk below
  `FALL_HEIGHT_RATIO` of its standing height above the terrain, or tilted
  past `MAX_TILT_GZ`. With feet-only contacts the trunk would otherwise pass
  through blocks;
- the critic also sees a clean height scan (Wojtek's grid, scaled), so
  value-loss regret can tell levels apart.
"""

import abc
import dataclasses
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

from config.hfield_config import HFieldConfig
from hfield.tiles import TileLayout
from wojtek_env.envs import height_scan
from wojtek_env.envs.base import resolve_backend

WOJTEK_HEIGHT = 0.10  # m, Wojtek's standing trunk height: scale 1
FALL_HEIGHT_RATIO = 0.6  # Wojtek: fall.min_height 0.06 of 0.10
MAX_TILT_GZ = -0.4  # Wojtek: fall.max_tilt_gz
WOJTEK_COMMAND = {"vx": (0.2, 0.6), "vy": (-0.2, 0.2), "wz": (-0.4, 0.4)}
SCAN_X, SCAN_Y, SCAN_N, SCAN_CLIP = (0.65, 1.45), (-0.3, 0.3), 5, 0.3

# pyramidal friction rows per contact are 4 (condim 3) to 10 (condim 6)
ROWS_PER_CONTACT = 6
CONTACTS_PER_GEOM = 2


def arena_scene(
    name: str,
    robot_xml: str,
    config: HFieldConfig,
    *,
    sensors: str,
    spawn_z: float,
    home_pose: str,
) -> str:
    """The robot's MJX model plus the arena's one `hfield` geom (`floor`)."""
    layout = TileLayout(config)
    r, amp, n = layout.half, config.amplitude, layout.nodes
    return f"""<mujoco model="{name}_accel_scene">
  <include file="{robot_xml}"/>
  <asset>
    <hfield name="hfield" nrow="{n}" ncol="{n}"
      size="{r:.4f} {r:.4f} {2 * amp:.6f} {config.base_z:.4f}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="hfield" pos="0 0 {-amp:.6f}"
      contype="1" conaffinity="0" priority="1" friction="1.0"/>
  </worldbody>
  {sensors}
  <keyframe>
    <key name="home" qpos="0 0 {spawn_z} 1 0 0 0 {home_pose}" ctrl="{home_pose}"/>
  </keyframe>
</mujoco>
"""


def compile_scene(xml: str, assets: dict[str, bytes]) -> Any:
    # mujoco is a C extension that ty cannot introspect
    return mujoco.MjModel.from_xml_string(  # ty: ignore[unresolved-attribute]
        xml, assets=assets
    )


def contact_budget(model: Any) -> tuple[int, int]:
    """(naconmax per env, njmax) from the geoms that collide with the hfield:
    two contacts each, `ROWS_PER_CONTACT` constraint rows per contact, plus
    one row per actuated joint limit. Go1 (4 feet) gets Playground's 8, 60."""
    # mujoco is a C extension that ty cannot introspect
    hfield_type = mujoco.mjtGeom.mjGEOM_HFIELD  # ty: ignore[unresolved-attribute]
    (hf,) = [i for i in range(model.ngeom) if model.geom_type[i] == hfield_type]
    contype, conaffinity = model.geom_contype, model.geom_conaffinity
    hits = (contype & conaffinity[hf]) | (conaffinity & contype[hf])
    colliding = int((hits != 0).sum()) - int(hits[hf] != 0)
    contacts = CONTACTS_PER_GEOM * colliding
    return contacts, model.nu + ROWS_PER_CONTACT * contacts


class TerrainRobot(abc.ABC):
    """Mixin over a Playground joystick env (put it first in the bases).

    A subclass provides the robot's constants, `scene_xml`, `assets`,
    `_build_env` (the Playground base `__init__`) and `_gravity_z`.
    """

    SCALE: ClassVar[float]  # standing trunk height / WOJTEK_HEIGHT
    STANDING_HEIGHT: ClassVar[float]  # m, trunk above flat ground, settled
    # (feet-only, full-collision) robot XML names for eval_sim2sim's
    # full_collision variant; None: the model has no full-collision version
    FULL_COLLISION: ClassVar[tuple[str, str] | None] = None
    # (model, rng) -> (model, in_axes); None: Playground has none for it
    domain_randomize: ClassVar[Callable[..., Any] | None] = None

    # the Playground env this mixes into provides these
    mj_model: Any
    mjx_model: Any
    _feet_site_id: Any
    _init_q: jax.Array

    @classmethod
    @abc.abstractmethod
    def scene_xml(cls, config: HFieldConfig) -> str: ...

    @classmethod
    @abc.abstractmethod
    def assets(cls) -> dict[str, bytes]: ...

    @abc.abstractmethod
    def _build_env(self, config: Any, hfield: HFieldConfig) -> None: ...

    @abc.abstractmethod
    def _gravity_z(self, data: mjx.Data) -> jax.Array:
        """z of gravity in the trunk frame: -1 upright, > -0.4 is a fall."""

    def __init__(
        self,
        hfield: HFieldConfig,
        backend: str = "jax",
        num_envs: int = 8192,
        episode_length: int = 1000,
    ):
        config = self.default_config()
        # every env shares the tiled arena's hfield_data, so warp is valid
        self._backend = resolve_backend(backend)
        config.impl = self._backend
        config.episode_length = episode_length
        self._build_env(config, hfield)
        self._hfield_config = hfield
        self._layout = TileLayout(hfield)
        contacts, config.njmax = contact_budget(self.mj_model)
        config.naconmax = contacts * num_envs
        # FunctionalEnv sets this to the env's tile centre around reset
        self.spawn_xy = jnp.zeros(2)
        self._scan_grid = jnp.array(
            height_scan.body_grid(
                [x * self.SCALE for x in SCAN_X],
                [y * self.SCALE for y in SCAN_Y],
                SCAN_N,
                SCAN_N,
            )
        )
        low = jnp.array([WOJTEK_COMMAND[k][0] for k in ("vx", "vy", "wz")])
        high = jnp.array([WOJTEK_COMMAND[k][1] for k in ("vx", "vy", "wz")])
        speed = jnp.array([self.SCALE, self.SCALE, 1.0])
        self._cmd_low, self._cmd_high = low * speed, high * speed

    @staticmethod
    @abc.abstractmethod
    def default_config() -> Any: ...

    def _write_scene(self, hfield: HFieldConfig) -> str:
        # Playground's env classes read the scene from a path; its includes
        # resolve through `assets`
        scene = Path(tempfile.mkdtemp()) / "accel_scene.xml"
        scene.write_text(self.scene_xml(hfield))
        return scene.as_posix()

    def terrain_height(self, xy: jax.Array) -> jax.Array:
        """Bilinear terrain height (m) at world `xy` (..., 2) of the live level."""
        c, layout = self._hfield_config, self._layout
        n = layout.nodes
        grid = self.mjx_model.hfield_data.reshape(n, n)
        u = (xy + layout.half) / (2 * layout.half) * (n - 1)
        u = jnp.clip(u, 0.0, n - 1.000001)
        i0 = jnp.floor(u).astype(jnp.int32)
        f = u - i0
        col, row = i0[..., 0], i0[..., 1]
        fx, fy = f[..., 0], f[..., 1]
        top = grid[row, col] * (1 - fx) + grid[row, col + 1] * fx
        bottom = grid[row + 1, col] * (1 - fx) + grid[row + 1, col + 1] * fx
        level = top * (1 - fy) + bottom * fy
        return (2 * level - 1) * c.amplitude

    def _forward_command(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(
            rng, (3,), minval=self._cmd_low, maxval=self._cmd_high
        )

    def sample_command(self, rng: jax.Array, *_: Any) -> jax.Array:
        return self._forward_command(rng)

    def _with_command(self, state: mjx_env.State, rng: jax.Array) -> mjx_env.State:
        """Envs whose reset does not draw through `sample_command` override."""
        del rng
        return state

    def _obs_dict(self, obs: Any) -> dict[str, jax.Array]:
        return dict(obs)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, cmd_rng = jax.random.split(rng)
        init_q = self._init_q
        self._init_q = init_q.at[0:2].add(self.spawn_xy)
        try:
            state = super().reset(rng)  # ty: ignore[unresolved-attribute]
        finally:
            self._init_q = init_q
        return self._with_scan(self._with_command(state, cmd_rng))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self._with_scan(
            super().step(state, action)  # ty: ignore[unresolved-attribute]
        )

    def _with_scan(self, state: mjx_env.State) -> mjx_env.State:
        data = state.data
        quat = data.qpos[3:7]
        xy = height_scan.world_xy(
            self._scan_grid, data.qpos[0:2], height_scan.yaw_from_quat(quat)
        )
        ref_z = jnp.min(data.site_xpos[self._feet_site_id][:, 2])
        scan = height_scan.scan_values(
            self.terrain_height(xy), ref_z, SCAN_CLIP * self.SCALE
        )
        obs = self._obs_dict(state.obs)
        obs["privileged_state"] = jnp.concatenate([obs["privileged_state"], scan])
        return dataclasses.replace(state, obs=obs)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        height = data.qpos[2] - self.terrain_height(data.qpos[0:2])
        low = height < FALL_HEIGHT_RATIO * self.STANDING_HEIGHT
        tilted = self._gravity_z(data) > MAX_TILT_GZ
        return low | tilted
