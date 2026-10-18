"""Unitree Go1 on the UED heightfield arena (MuJoCo Playground joystick task).

Playground's `Go1JoystickRoughTerrain` (feet-only collisions, its own reward
and blind actor) with four changes so Go1 runs the same benchmark as Wojtek:

- the scene is the tiled ACCEL arena (`hfield.tiles`): one swappable
  `hfield` geom (`floor`, geom 0 so Playground's friction DR still hits it)
  sized by `HFieldConfig`; the robot spawns on `spawn_xy`, its tile centre;
- commands walk forward, `GO1_SCALE` times Wojtek's range, so the robot
  leaves the spawn pad and meets the level within an episode;
- a fall is Wojtek's rule at Go1's size: trunk below `FALL_HEIGHT_RATIO` of
  standing height above the terrain, or tilted past `MAX_TILT_GZ`. With
  feet-only contacts the trunk would otherwise pass through blocks;
- the critic also sees a clean height scan (Wojtek's grid, scaled), so
  value-loss regret can tell levels apart.
"""

import dataclasses
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick

from config.hfield_config import HFieldConfig
from hfield.tiles import TileLayout
from wojtek_env.envs import height_scan
from wojtek_env.envs.base import resolve_backend

# Go1 standing trunk height (0.258 m, measured) over Wojtek's design 0.10 m
GO1_SCALE = 2.5
STANDING_HEIGHT = 0.258
FALL_HEIGHT_RATIO = 0.6  # Wojtek: fall.min_height 0.06 of 0.10
MAX_TILT_GZ = -0.4  # Wojtek: fall.max_tilt_gz
WOJTEK_COMMAND = {"vx": (0.2, 0.6), "vy": (-0.2, 0.2), "wz": (-0.4, 0.4)}
SCAN_X, SCAN_Y, SCAN_N, SCAN_CLIP = (0.65, 1.45), (-0.3, 0.3), 5, 0.3

HOME_POSE = "0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"


def scene_xml(config: HFieldConfig) -> str:
    layout = TileLayout(config)
    r, amp, n = layout.half, config.amplitude, layout.nodes
    return f"""<mujoco model="go1_accel_scene">
  <include file="go1_mjx_feetonly.xml"/>
  <asset>
    <hfield name="hfield" nrow="{n}" ncol="{n}"
      size="{r:.4f} {r:.4f} {2 * amp:.6f} {config.base_z:.4f}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="hfield" pos="0 0 {-amp:.6f}"
      contype="1" conaffinity="0" priority="1" friction="1.0"/>
  </worldbody>
  <include file="sensor_feet.xml"/>
  <keyframe>
    <key name="home" qpos="0 0 0.35 1 0 0 0 {HOME_POSE}" ctrl="{HOME_POSE}"/>
  </keyframe>
</mujoco>
"""


class Go1Terrain(joystick.Joystick):
    def __init__(
        self,
        hfield: HFieldConfig,
        backend: str = "jax",
        num_envs: int = 8192,
        episode_length: int = 1000,
    ):
        config = joystick.default_config()
        # every env shares the tiled arena's hfield_data, so warp is valid
        self._backend = resolve_backend(backend)
        config.impl = self._backend
        config.episode_length = episode_length
        # Playground's rough-terrain Go1 budget: 8 contacts per env, pooled
        config.naconmax = 8 * num_envs
        config.njmax = 12 + 48
        # Go1Env reads the scene from a path; its includes resolve via assets
        scene = Path(tempfile.mkdtemp()) / "go1_accel_scene.xml"
        scene.write_text(scene_xml(hfield))
        go1_base.Go1Env.__init__(self, xml_path=scene.as_posix(), config=config)
        self._post_init()
        self._hfield_config = hfield
        self._layout = TileLayout(hfield)
        # FunctionalEnv sets this to the env's tile centre around reset
        self.spawn_xy = jnp.zeros(2)
        self._scan_grid = jnp.array(
            height_scan.body_grid(
                [x * GO1_SCALE for x in SCAN_X],
                [y * GO1_SCALE for y in SCAN_Y],
                SCAN_N,
                SCAN_N,
            )
        )
        self._cmd_low = jnp.array([WOJTEK_COMMAND[k][0] for k in ("vx", "vy", "wz")])
        self._cmd_high = jnp.array([WOJTEK_COMMAND[k][1] for k in ("vx", "vy", "wz")])
        speed = jnp.array([GO1_SCALE, GO1_SCALE, 1.0])
        self._cmd_low, self._cmd_high = self._cmd_low * speed, self._cmd_high * speed

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

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        del x_k
        return self._forward_command(rng)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, cmd_rng = jax.random.split(rng)
        init_q = self._init_q
        self._init_q = init_q.at[0:2].add(self.spawn_xy)
        try:
            state = super().reset(rng)
        finally:
            self._init_q = init_q
        state.info["command"] = self._forward_command(cmd_rng)
        return self._with_scan(
            dataclasses.replace(state, obs=self._get_obs(state.data, state.info))
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        return self._with_scan(super().step(state, action))

    def _with_scan(self, state: mjx_env.State) -> mjx_env.State:
        data = state.data
        quat = data.qpos[3:7]
        xy = height_scan.world_xy(
            self._scan_grid, data.qpos[0:2], height_scan.yaw_from_quat(quat)
        )
        ref_z = jnp.min(data.site_xpos[self._feet_site_id][:, 2])
        scan = height_scan.scan_values(
            self.terrain_height(xy), ref_z, SCAN_CLIP * GO1_SCALE
        )
        obs = dict(state.obs)
        obs["privileged_state"] = jnp.concatenate([obs["privileged_state"], scan])
        return dataclasses.replace(state, obs=obs)

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        height = data.qpos[2] - self.terrain_height(data.qpos[0:2])
        low = height < FALL_HEIGHT_RATIO * STANDING_HEIGHT
        tilted = self.get_gravity(data)[-1] > MAX_TILT_GZ
        return low | tilted
