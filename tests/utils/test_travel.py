"""Lane isolation budget: how far from its tile centre an env reads terrain."""

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import pytest

from hfield.tiles import TileLayout
from scripts.brax_ued import hfield_config, parse_args, run_spec
from utils.env import TravelBounds, max_command_speed, travel_bounds
from wojtek_env.envs import registry
from wojtek_env.terrain.terrain_env import SPAWN_EDGE_MARGIN

SCAN = math.hypot(1.45, 0.3)


def profile(name: str) -> tuple[TravelBounds, TileLayout]:
    args = parse_args(["--mode", "accel", "--name", "t", "--profile", name])
    config = registry.TASKS["joystick"][1]()
    registry._apply_overrides(config, run_spec(args)["env_overrides"])
    base = SimpleNamespace(_config=config, dt=config.ctrl_dt)
    return travel_bounds(base), TileLayout(hfield_config(args))


def test_deploy_speed_is_the_command_box_corner() -> None:
    travel, _ = profile("deploy")
    assert travel.speed == pytest.approx(math.hypot(1.2, 0.5))


def test_benchmark_speed_is_the_fast_walk() -> None:
    travel, _ = profile("benchmark")
    assert travel.speed == pytest.approx(1.2)


@pytest.mark.parametrize("name", ["benchmark", "deploy"])
def test_critic_reads_the_scan_actor_only_its_feet(name: str) -> None:
    travel, _ = profile(name)
    assert travel.sensing == pytest.approx(SCAN)
    assert travel.actor_sensing == pytest.approx(SPAWN_EDGE_MARGIN)


@pytest.mark.parametrize("name", ["benchmark", "deploy"])
def test_a_300_step_level_stays_on_its_tile(name: str) -> None:
    travel, layout = profile(name)
    layout.check_reach(travel.reach(300, travel.sensing), "300 steps")


def test_deploy_scan_reaches_past_an_8_m_arena() -> None:
    travel, layout = profile("deploy")
    eight = TileLayout(dataclasses.replace(layout.config, reach=8.0))
    reach = travel.reach(300, travel.sensing)
    assert reach == pytest.approx(0.15 + 6 * math.hypot(1.2, 0.5) + SCAN)
    with pytest.raises(ValueError, match=r"HFieldConfig.reach >= 10"):
        eight.check_reach(reach, "300 steps")


def test_1000_step_levels_are_refused() -> None:
    travel, layout = profile("deploy")
    with pytest.raises(ValueError, match="1000 steps"):
        layout.check_reach(travel.reach(1000, travel.sensing), "1000 steps")


def test_disabled_modes_do_not_count() -> None:
    command: Any = registry.TASKS["joystick"][1]().command
    assert max_command_speed(command) == pytest.approx(math.hypot(0.6, 0.4))


def test_feature_spawns_count_the_whole_patch() -> None:
    config = registry.TASKS["joystick"][1]()
    config.terrain.spawn_mode = "feature"
    base = SimpleNamespace(
        _config=config, dt=config.ctrl_dt, _terrain=SimpleNamespace(tile_size=6.0)
    )
    assert travel_bounds(base).jitter == pytest.approx(3.0 - SPAWN_EDGE_MARGIN)
