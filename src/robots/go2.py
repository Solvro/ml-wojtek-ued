"""Unitree Go2: Go1's joystick env on the menagerie Go2 model, rewritten
feet-only (docs/robots.md)."""

import re
from typing import Any

from mujoco import mjx
from mujoco_playground._src import mjx_env

from config.hfield_config import HFieldConfig
from robots.go1 import Go1Terrain
from robots.terrain import PDGains, arena_scene, compile_scene

FEET = ("FL", "FR", "RL", "RR")
HOME_POSE = "0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"
FEET_ONLY = "go2_feetonly.xml"
# menagerie's kp 50 x2, kv and damping x sqrt(2) (same damping ratio):
# menagerie's own gains sag the trunk and track swing targets loosely
GAINS = PDGains(kp=100.0, kd=0.707, damping=2.828)
SENSORS = "\n".join(
    [
        "<sensor>",
        '<framezaxis objtype="site" objname="imu" name="upvector"/>',
        '<velocimeter site="imu" name="local_linvel"/>',
        *(
            f'<framelinvel objtype="site" objname="{f}" '
            f'name="{f}_global_linvel"/>'
            for f in FEET
        ),
        *(
            f'<contact name="{f}_floor_found" geom1="{f}" geom2="floor"'
            ' reduce="mindist" num="1" data="found"/>'
            for f in FEET
        ),
        "</sensor>",
    ]
)


def replace_once(xml: str, old: str, new: str) -> str:
    if xml.count(old) != 1:
        raise ValueError(
            f"menagerie go2_mjx.xml has changed: expected exactly one {old!r}; "
            "update robots/go2.py feet_only_model to the new file"
        )
    return xml.replace(old, new)


def feet_only_model(xml: str) -> str:
    xml = replace_once(
        xml,
        '<default class="collision">\n        <geom group="3"/>',
        '<default class="collision">\n        '
        '<geom group="3" contype="0" conaffinity="0"/>',
    )
    xml = replace_once(
        xml,
        '<geom size="0.0175" pos="-0.002 0 -0.213" priority="1"',
        # MJX: no hfield-sphere margin
        '<geom size="0.0175" pos="-0.002 0 -0.213" contype="0" conaffinity="1" '
        'margin="0" priority="1"',
    )
    xml = replace_once(xml, '<body name="base"', '<body name="trunk"')
    xml = re.sub(r'<site name="(\w+)_foot"', r'<site name="\1"', xml)
    # the scene's keyframe starts the robot above the arena
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.DOTALL)
    return replace_once(xml, "</mujoco>", f"{SENSORS}\n</mujoco>")


class Go2Terrain(Go1Terrain):
    # standing trunk height 0.270 m (measured) over Wojtek's design 0.10 m
    SCALE = 2.7
    STANDING_HEIGHT = 0.270
    FULL_COLLISION = None

    @classmethod
    def scene_xml(cls, config: HFieldConfig) -> str:
        return arena_scene(
            "go2",
            FEET_ONLY,
            config,
            sensors="",
            spawn_z=0.33,
            home_pose=HOME_POSE,
        )

    @classmethod
    def assets(cls) -> dict[str, bytes]:
        assets: dict[str, bytes] = {}
        path = mjx_env.MENAGERIE_PATH / "unitree_go2"
        mjx_env.update_assets(assets, path, "*.xml")
        mjx_env.update_assets(assets, path / "assets")
        robot = feet_only_model(assets["go2_mjx.xml"].decode())
        return {**assets, FEET_ONLY: robot.encode()}

    def _build_env(self, config: Any, hfield: HFieldConfig) -> None:
        mjx_env.MjxEnv.__init__(self, config, None)
        self._xml_path = self._write_scene(hfield)
        self._model_assets = self.assets()
        model = compile_scene(self.scene_xml(hfield), self._model_assets)
        # Go1Env's setup with Go2's own gains: Go1's kp 35 sags the heavier
        # Go2 well below its keyframe
        GAINS.apply(model)
        model.opt.timestep = config.sim_dt
        model.opt.ccd_iterations = 20
        model.vis.global_.offwidth = 3840
        model.vis.global_.offheight = 2160
        self._mj_model = model
        self._mjx_model = mjx.put_model(model, impl=config.impl)
        self._imu_site_id = model.site("imu").id
        self._feet_floor_found_sensor = [
            model.sensor(f"{foot}_floor_found").id for foot in FEET
        ]
        self._post_init()
