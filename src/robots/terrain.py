"""Shared base of the Playground quadrupeds on the tiled UED arena
(docs/robots.md)."""

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

WOJTEK_HEIGHT = 0.10  # m, Wojtek's standing trunk height: scale 1
FALL_HEIGHT_RATIO = 0.6  # Wojtek: fall.min_height 0.06 of 0.10
MAX_TILT_GZ = -0.4  # Wojtek: fall.max_tilt_gz
# eval commands are at least this much faster than the success threshold
EVAL_SPEED_MARGIN = 1.2
SCAN_X, SCAN_Y, SCAN_N, SCAN_CLIP = (0.65, 1.45), (-0.3, 0.3), 5, 0.3


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
"""  # noqa: E501 (MJCF template: one element per line)


def compile_scene(xml: str, assets: dict[str, bytes]) -> Any:
    # mujoco is a C extension that ty cannot introspect
    return mujoco.MjModel.from_xml_string(  # ty: ignore[unresolved-attribute]
        xml, assets=assets
    )


def playground_backend(backend: str) -> str:
    """jax: on warp, a foot's contact with the tiled arena's hfield blows up
    more often the farther its tile is from the arena origin (16 x 16 tiles:
    1.5-5 % of random-action steps for Go1, 0-0.5 % on a 2 x 2 arena), which
    ends those episodes and teaches the robot to stand still."""
    if backend == "warp":
        raise ValueError(
            "backend=warp is not supported for the Playground robots: their "
            "foot contacts with the tiled arena diverge on warp far from the "
            "arena origin (docs/robots.md). Use backend=jax or "
            "backend=auto (jax for these robots)"
        )
    if backend not in ("auto", "jax"):
        raise ValueError(f"backend must be jax, warp or auto, got {backend!r}")
    return "jax"


class TerrainRobot(abc.ABC):
    """Mixin over a Playground joystick env; put it first in the bases."""

    SCALE: ClassVar[float]  # standing trunk height / WOJTEK_HEIGHT
    STANDING_HEIGHT: ClassVar[float]  # m, trunk above flat ground, settled
    # (feet-only, full-collision) robot XML names for sim2sim's
    # full_collision variant; None: the model has no full-collision version
    FULL_COLLISION: ClassVar[tuple[str, str] | None] = None
    # (model, rng) -> (model, in_axes); None: Playground has none for it
    domain_randomize: ClassVar[Callable[..., Any] | None] = None

    # the Playground env this mixes into provides these
    mj_model: Any
    mjx_model: Any
    dt: Any
    _config: Any
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
        episode_length: int = 1000,
        eval_travel: tuple[float, int] | None = None,
    ):
        """`eval_travel`: (success threshold in m, steps per episode) of an
        eval env; its vx commands start where the threshold is reachable."""
        config = self.default_config()
        self._backend = playground_backend(backend)
        config.impl = self._backend
        config.episode_length = episode_length
        self._build_env(config, hfield)
        self._hfield_config = hfield
        self._layout = TileLayout(hfield)
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
        low, high = self.native_command_range()
        # the forward slice of Playground's (vx, vy, wz) box
        low[0] = max(low[0], 0.0)
        if eval_travel is not None:
            meters, steps = eval_travel
            low[0] = max(low[0], EVAL_SPEED_MARGIN * meters / (steps * self.dt))
        if low[0] >= high[0]:
            raise ValueError(
                f"{type(self).__name__}: eval vx floor {low[0]:.2f} m/s is at "
                f"or past the top of its command range {high[0]:.2f} m/s; "
                "lower min_travel or raise level_steps"
            )
        self._cmd_low, self._cmd_high = jnp.array(low), jnp.array(high)

    @staticmethod
    @abc.abstractmethod
    def default_config() -> Any: ...

    @abc.abstractmethod
    def native_command_range(self) -> tuple[list[float], list[float]]:
        """(low, high) of (vx, vy, wz) in Playground's config of the env."""

    def _write_scene(self, hfield: HFieldConfig) -> str:
        # Playground's env classes read the scene from a path; its includes
        # resolve through `assets`
        scene = Path(tempfile.mkdtemp()) / "accel_scene.xml"
        scene.write_text(self.scene_xml(hfield))
        return scene.as_posix()

    def terrain_height(self, xy: jax.Array) -> jax.Array:
        """Bilinear height (m) of the live arena at world `xy` (..., 2)."""
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

    def command_frame_velocity(self, data: mjx.Data) -> jax.Array:
        """Linear velocity in the frame the reward tracks commands in: the
        `local_linvel` velocimeter's site (Barkour's faces the torso's -x)."""
        return mjx_env.get_sensor_data(self.mj_model, data, "local_linvel")

    def _forward_command(self, rng: jax.Array) -> jax.Array:
        return jax.random.uniform(
            rng, (3,), minval=self._cmd_low, maxval=self._cmd_high
        )

    def sample_command(self, rng: jax.Array, *_: Any) -> jax.Array:
        return self._forward_command(rng)

    def _with_command(
        self, state: mjx_env.State, rng: jax.Array
    ) -> mjx_env.State:
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
        obs["privileged_state"] = jnp.concatenate(
            [obs["privileged_state"], scan]
        )
        return dataclasses.replace(state, obs=obs)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        height = data.qpos[2] - self.terrain_height(data.qpos[0:2])
        low = height < FALL_HEIGHT_RATIO * self.STANDING_HEIGHT
        tilted = self._gravity_z(data) > MAX_TILT_GZ
        return low | tilted
