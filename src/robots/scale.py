import dataclasses

from config.hfield_config import HFieldConfig
from config.schema import Robot, RobotConfig
from robots.registry import TERRAIN_ROBOTS
from robots.terrain import OWN_GAINS, PDGains

# each robot's own size relative to Wojtek; `robot.scale` overrides it
ROBOT_SCALE = {
    "wojtek": 1.0,
    **{name: robot.SCALE for name, robot in TERRAIN_ROBOTS.items()},
}


def default_robot(name: Robot) -> RobotConfig:
    """`name` with its own gains and scale, for runs logged by name only."""
    return RobotConfig(name=name, kp=None, kd=None, damping=None, scale=None)


def robot_scale(robot: RobotConfig) -> float:
    if robot.scale is None:
        return ROBOT_SCALE[robot.name]
    if robot.name not in TERRAIN_ROBOTS:
        raise ValueError(
            f"robot.scale={robot.scale} applies to the Playground robots; "
            f"robot={robot.name} is the scale-1 reference. Drop the override"
        )
    if robot.scale <= 0:
        raise ValueError(f"robot.scale={robot.scale} must be > 0")
    return robot.scale


def pd_gains(robot: RobotConfig) -> PDGains:
    gains = PDGains(kp=robot.kp, kd=robot.kd, damping=robot.damping)
    if gains != OWN_GAINS and robot.name not in TERRAIN_ROBOTS:
        raise ValueError(
            f"robot.kp/kd/damping apply to the Playground robots; "
            f"robot={robot.name} sets its actuators in wojtek_env. Drop "
            "the overrides"
        )
    return gains


def hfield_config(robot: RobotConfig, arena: HFieldConfig) -> HFieldConfig:
    """The `arena` config (Wojtek units) at the robot's size: every length
    scales, slopes do not."""
    s = robot_scale(robot)
    return dataclasses.replace(
        arena,
        patch_radius=arena.patch_radius * s,
        cell_size=arena.cell_size * s,
        amplitude=arena.amplitude * s,
        pad_radius=arena.pad_radius * s,
        pad_taper=arena.pad_taper * s,
        reach=arena.reach * s,
    )


def min_travel(robot: RobotConfig, wojtek_m: float) -> float:
    return wojtek_m * robot_scale(robot)
