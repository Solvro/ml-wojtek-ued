"""Boston Dynamics Spot: Playground's SpotJoystick on the tiled arena."""

from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import joystick

from config.hfield_config import HFieldConfig
from robots.terrain import TerrainRobot, arena_scene

HOME_POSE = "0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8 0 1.04 -1.8"


class SpotTerrain(TerrainRobot, joystick.Joystick):
    # standing trunk height 0.433 m (measured) over Wojtek's design 0.10 m
    SCALE = 4.3
    STANDING_HEIGHT = 0.433
    FULL_COLLISION = ("spot_mjx_feetonly.xml", "spot_mjx.xml")

    def __init__(
        self,
        hfield: HFieldConfig,
        backend: str = "jax",
        episode_length: int = 1000,
    ):
        super().__init__(hfield, backend, episode_length)
        # Wojtek's range x 4.3 reaches vx 2.6 m/s, beyond what Playground's
        # Spot was tuned for (real Spot: ~1.6 m/s): cap at its command range
        c = self._config.command_config
        keys = ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")
        low = jnp.array([c[k][0] for k in keys])
        high = jnp.array([c[k][1] for k in keys])
        self._cmd_low = jnp.clip(self._cmd_low, low, high)
        self._cmd_high = jnp.clip(self._cmd_high, low, high)

    @classmethod
    def scene_xml(cls, config: HFieldConfig) -> str:
        return arena_scene(
            "spot",
            "spot_mjx_feetonly.xml",
            config,
            sensors='<include file="sensor.xml"/>',
            spawn_z=0.5,
            home_pose=HOME_POSE,
        )

    @classmethod
    def assets(cls) -> dict[str, bytes]:
        return spot_base.get_assets()

    @staticmethod
    def default_config() -> Any:
        return joystick.default_config()

    def _build_env(self, config: Any, hfield: HFieldConfig) -> None:
        spot_base.SpotEnv.__init__(
            self, xml_path=self._write_scene(hfield), config=config
        )
        self._post_init()
        self._pert_func = (
            self._maybe_apply_perturbation
            if config.pert_config.enable
            else lambda state, _: state
        )

    def _gravity_z(self, data: mjx.Data) -> jax.Array:
        # Playground's Spot "gravity" is the trunk's up vector: +1 upright
        return -self.get_gravity(data)[-1]
