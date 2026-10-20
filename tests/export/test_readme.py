from collections.abc import Callable, Iterator
from typing import Any

import pytest

from config.hfield_config import HFieldConfig
from config.schema import Robot, TrainConfig
from env.functional import FunctionalEnv
from env.overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
    terrain_task_overrides,
)
from export.contract import Contract, build_contract, index_ranges
from export.readme import render_readme
from export.verify import Verification
from hfield.arena import write_accel_arena
from hfield.tiles import TileLayout
from robots.registry import TERRAIN_ROBOTS
from robots.scale import hfield_config
from training.summary import TrainingSummary
from wojtek_env import paths

Compose = Callable[..., Any]

ARENA = HFieldConfig(
    patch_radius=1.5,
    amplitude=0.3,
    pad_radius=0.55,
    pad_taper=0.2,
    tiles=2,
    reach=2.0,
)
OBS_DIMS = {"wojtek": 54, "go1": 48, "go2": 48, "spot": 81, "barkour": 465}
VERIFICATION = Verification(256, 3e-7, 400, 4e-7)
SUMMARY = TrainingSummary(
    env_steps=1500,
    wall_time_s=7200.0,
    window_iterations=3,
    reward_per_step=0.25,
    train_success=0.5,
    falls_per_step=0.001,
    forward_vel=0.3,
    cmd_vx=0.4,
)


def wojtek_env() -> Iterator[FunctionalEnv]:
    write_accel_arena(ARENA, kind="test")
    overrides = merge_overrides(
        accel_arena_overrides(ARENA.pad_radius, "jax", "test"),
        movement_overrides(),
        terrain_overrides(),
        terrain_task_overrides(),
        {"sim": {"num_envs": TileLayout(ARENA).n_tiles}},
    )
    yield FunctionalEnv(task="joystick", env_overrides=overrides)
    for path in paths.terrain_paths("test").values():
        path.unlink(missing_ok=True)


def playground_env(robot: str) -> FunctionalEnv:
    config = hfield_config(robot, ARENA)  # ty: ignore[invalid-argument-type]
    return FunctionalEnv(
        robot=TERRAIN_ROBOTS[robot](config, "jax"), domain_randomize=True
    )


@pytest.fixture(scope="module", params=list(Robot), ids=lambda r: r.value)
def built(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[Robot, Contract, FunctionalEnv]]:
    robot: Robot = request.param
    if robot == Robot.wojtek:
        if not paths.ROBOT_XML.exists():
            pytest.skip("robot model not built; run `just build`")
        gen = wojtek_env()
        env = next(gen)
        yield robot, build_contract(robot, env), env
        next(gen, None)
    else:
        env = playground_env(robot.value)
        yield robot, build_contract(robot, env), env


def test_the_contract_covers_the_whole_actor_observation(
    built: tuple[Robot, Contract, FunctionalEnv],
) -> None:
    robot, contract, env = built
    assert contract.obs_dim == env.observation_size.actor == OBS_DIMS[robot]
    assert contract.action_dim == env.action_size == 12
    assert contract.ctrl_dt == pytest.approx(0.02)


def test_joint_tables_are_complete_and_ordered(
    built: tuple[Robot, Contract, FunctionalEnv],
) -> None:
    _, contract, _ = built
    names = [j.name for j in contract.joints]
    assert len(set(names)) == 12
    for joint in contract.joints:
        assert joint.kp > 0 and joint.kd >= 0
        assert joint.target_min < joint.target_max
        assert joint.action_scale > 0


def test_readme_lists_every_component_with_its_index_range(
    built: tuple[Robot, Contract, FunctionalEnv], compose_cfg: Compose
) -> None:
    robot, contract, _ = built
    cfg: TrainConfig = compose_cfg(
        "train", "mode=dr", "name=t", f"robot={robot.value}"
    )
    readme = render_readme("t", cfg, contract, VERIFICATION, SUMMARY)
    for component, start, stop in index_ranges(contract):
        row = f"| `{component.name}` | {start}:{stop} | {component.size} |"
        assert row in readme
    assert f"`[batch, {contract.obs_dim}]`" in readme
    for joint in contract.joints:
        assert f"`{joint.name}`" in readme
    assert "```python" in readme and "onnxruntime" in readme
    assert "reward per step" in readme


def test_wojtek_readme_has_the_gait_clock_and_the_action_filter(
    built: tuple[Robot, Contract, FunctionalEnv], compose_cfg: Compose
) -> None:
    robot, contract, _ = built
    if robot != Robot.wojtek:
        pytest.skip("Wojtek only")
    cfg = compose_cfg("train", "mode=dr", "name=t")
    readme = render_readme("t", cfg, contract, VERIFICATION, None)
    assert "## Gait clock" in readme
    assert "gait_phase_features" in readme
    assert "FILTER = 0.2" in readme
    assert "Not recorded" in readme
    assert contract.stance is not None
    assert contract.stance.height_index == 3 + 3 + 12 + 12 + 12 + 3
