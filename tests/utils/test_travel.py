"""Lane isolation budget: how far from its tile centre an env reads terrain."""

import dataclasses
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from hfield.tiles import TileLayout
from scripts.brax_ued import hfield_config, wojtek_overrides
from utils.env import TravelBounds, max_command_speed, travel_bounds
from wojtek_env.envs import registry
from wojtek_env.terrain.terrain_env import SPAWN_EDGE_MARGIN

SCAN = math.hypot(1.45, 0.3)
Compose = Callable[..., Any]


@pytest.fixture
def benchmark(compose_cfg: Compose) -> tuple[TravelBounds, TileLayout]:
    cfg = compose_cfg("train", "mode=accel", "name=t")
    arena = hfield_config(cfg.robot.name, cfg.arena)
    config = registry.TASKS["joystick"][1]()
    overrides = wojtek_overrides(arena, cfg.backend, 1, train=True)
    registry._apply_overrides(config, overrides)
    base = SimpleNamespace(_config=config, dt=config.ctrl_dt)
    return travel_bounds(base), TileLayout(arena)


def test_benchmark_speed_is_the_fast_walk(
    benchmark: tuple[TravelBounds, TileLayout],
) -> None:
    travel, _ = benchmark
    assert travel.speed == pytest.approx(1.2)


def test_critic_reads_the_scan_actor_only_its_feet(
    benchmark: tuple[TravelBounds, TileLayout],
) -> None:
    travel, _ = benchmark
    assert travel.sensing == pytest.approx(SCAN)
    assert travel.actor_sensing == pytest.approx(SPAWN_EDGE_MARGIN)


def test_a_300_step_level_stays_on_its_tile(
    benchmark: tuple[TravelBounds, TileLayout],
) -> None:
    travel, layout = benchmark
    layout.check_reach(travel.reach(300, travel.sensing), "300 steps")


def test_scan_reaches_past_an_8_m_arena(
    benchmark: tuple[TravelBounds, TileLayout],
) -> None:
    travel, layout = benchmark
    eight = TileLayout(dataclasses.replace(layout.config, reach=8.0))
    reach = travel.reach(300, travel.sensing)
    assert reach == pytest.approx(0.15 + 6 * 1.2 + SCAN)
    with pytest.raises(ValueError, match=r"HFieldConfig.reach >= 9"):
        eight.check_reach(reach, "300 steps")


def test_1000_step_levels_are_refused(
    benchmark: tuple[TravelBounds, TileLayout],
) -> None:
    travel, layout = benchmark
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
