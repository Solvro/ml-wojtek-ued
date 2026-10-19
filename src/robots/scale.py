import dataclasses

from config.hfield_config import HFieldConfig
from config.schema import Robot
from robots.registry import TERRAIN_ROBOTS

ROBOT_SCALE = {
    "wojtek": 1.0,
    **{name: robot.SCALE for name, robot in TERRAIN_ROBOTS.items()},
}


def hfield_config(robot: Robot, arena: HFieldConfig) -> HFieldConfig:
    """The `arena` config (Wojtek units) at the robot's size: every length
    scales, slopes do not."""
    s = ROBOT_SCALE[robot]
    return dataclasses.replace(
        arena,
        patch_radius=arena.patch_radius * s,
        cell_size=arena.cell_size * s,
        amplitude=arena.amplitude * s,
        pad_radius=arena.pad_radius * s,
        pad_taper=arena.pad_taper * s,
        reach=arena.reach * s,
    )


def min_travel(robot: Robot, wojtek_m: float) -> float:
    return wojtek_m * ROBOT_SCALE[robot]
