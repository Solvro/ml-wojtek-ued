import math
from typing import Any, NamedTuple

from robots.terrain import SCAN_X, SCAN_Y, TerrainRobot
from wojtek_env.terrain.terrain_env import SPAWN_EDGE_MARGIN


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


def terrain_robot_bounds(robot: Any) -> TravelBounds:
    low, high = robot._cmd_low, robot._cmd_high
    speed = math.hypot(
        float(max(abs(low[0]), abs(high[0]))),
        float(max(abs(low[1]), abs(high[1]))),
    )
    feet = robot.SCALE * SPAWN_EDGE_MARGIN
    # the scan is critic-only
    sensing = max(feet, robot.SCALE * scan_reach(SCAN_X, SCAN_Y))
    return TravelBounds(0.0, speed, sensing, feet, robot.dt)


def travel_bounds(base: Any) -> TravelBounds:
    if isinstance(base, TerrainRobot):
        return terrain_robot_bounds(base)
    c = base._config
    hs = c.height_scan
    scan = scan_reach(hs.x_range, hs.y_range) if hs.enable else 0.0
    actor_scan = any(k.startswith("height_scan") for k in c.obs.state)
    jitter = c.terrain.pad_jitter
    if c.terrain.spawn_mode != "pad":
        # feature spawns land anywhere inside the patch
        jitter = base._terrain.tile_size / 2 - SPAWN_EDGE_MARGIN
    return TravelBounds(
        jitter=float(jitter),
        speed=max_command_speed(c.command),
        sensing=max(SPAWN_EDGE_MARGIN, scan),
        actor_sensing=max(SPAWN_EDGE_MARGIN, scan if actor_scan else 0.0),
        dt=base.dt,
    )
