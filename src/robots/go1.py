"""Unitree Go1 on the UED heightfield arena (MuJoCo Playground joystick task).

Playground's `Go1JoystickRoughTerrain` (feet-only collisions, its own reward
and blind actor) on the tiled ACCEL arena; the arena, command, fall and
critic-scan changes are `robots.terrain.TerrainRobot`'s.
"""

import dataclasses
from typing import Any

import jax
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground._src.locomotion.go1 import randomize as go1_randomize

from config.hfield_config import HFieldConfig
from robots.terrain import TerrainRobot, arena_scene

HOME_POSE = "0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"


class Go1Terrain(TerrainRobot, joystick.Joystick):
    # standing trunk height 0.258 m (measured) over Wojtek's design 0.10 m
    SCALE = 2.5
    STANDING_HEIGHT = 0.258
    FULL_COLLISION = ("go1_mjx_feetonly.xml", "go1_mjx_fullcollisions.xml")
    domain_randomize = staticmethod(go1_randomize.domain_randomize)

    @classmethod
    def scene_xml(cls, config: HFieldConfig) -> str:
        return arena_scene(
            "go1",
            "go1_mjx_feetonly.xml",
            config,
            sensors='<include file="sensor_feet.xml"/>',
            spawn_z=0.35,
            home_pose=HOME_POSE,
        )

    @classmethod
    def assets(cls) -> dict[str, bytes]:
        return go1_base.get_assets()

    @staticmethod
    def default_config() -> Any:
        return joystick.default_config()

    def _build_env(self, config: Any, hfield: HFieldConfig) -> None:
        xml_path = self._write_scene(hfield)
        go1_base.Go1Env.__init__(self, xml_path=xml_path, config=config)
        self._post_init()

    def _gravity_z(self, data: mjx.Data) -> jax.Array:
        return self.get_gravity(data)[-1]

    def _with_command(self, state: mjx_env.State, rng: jax.Array) -> mjx_env.State:
        # Playground's reset draws a command of its own range, not
        # `sample_command`; the obs it built holds that one
        state.info["command"] = self._forward_command(rng)
        obs = self._get_obs(state.data, state.info)
        return dataclasses.replace(state, obs=obs)
