"""Boston Dynamics Spot on the UED heightfield arena (MuJoCo Playground).

Playground's `SpotJoystick` (feet-only collisions, its own reward and blind
actor); the arena, command, fall and critic-scan changes are
`robots.terrain.TerrainRobot`'s. Playground has no Spot randomization, so DR
is off for it.
"""

from typing import Any

import jax
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
