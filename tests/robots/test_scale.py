"""`robot.kp` / `kd` / `damping` / `scale` reach the model, the arena and
the success threshold; null keeps each robot's own values."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from config.schema import Backend, Robot
from env.factory import make_env
from robots.registry import TERRAIN_ROBOTS
from robots.scale import (
    ROBOT_SCALE,
    default_robot,
    hfield_config,
    min_travel,
    robot_scale,
)

Compose = Callable[..., Any]
ARENA = HFieldConfig(
    patch_radius=1.5,
    amplitude=0.3,
    pad_radius=0.55,
    pad_taper=0.2,
    tiles=2,
    reach=2.0,
)


def actuators(model: Any) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(model.actuator_gainprm[:, 0]),
        np.asarray(model.actuator_biasprm[:, 1]),
        np.asarray(model.actuator_biasprm[:, 2]),
        np.asarray(model.dof_damping[6:]),
    )


def env_base(robot: Any) -> Any:
    config = hfield_config(robot, ARENA)
    env = make_env(robot, Backend.jax, config, num_envs=1, train=False)
    return env._env.unwrapped


@pytest.mark.parametrize("robot", ["go1", "go2", "barkour"])
def test_robot_defaults_are_null(compose_cfg: Compose, robot: str) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", f"robot={robot}")
    assert (cfg.robot.kp, cfg.robot.kd, cfg.robot.damping) == (None,) * 3
    assert cfg.robot.scale is None
    assert robot_scale(cfg.robot) == TERRAIN_ROBOTS[robot].SCALE


def test_gain_overrides_reach_mj_and_mjx_models(compose_cfg: Compose) -> None:
    cfg = compose_cfg(
        "eval",
        "checkpoint=c.pkl",
        "name=t",
        "out_dir=o",
        "robot=go2",
        "robot.kp=70",
        "robot.kd=0.8",
        "robot.damping=1.5",
    )
    base = env_base(cfg.robot)
    for model in (base.mj_model, base.mjx_model):
        kp, bias_kp, bias_kd, damping = actuators(model)
        np.testing.assert_allclose(kp, 70.0)
        np.testing.assert_allclose(bias_kp, -70.0)
        np.testing.assert_allclose(bias_kd, -0.8)
        np.testing.assert_allclose(damping, 1.5)


def test_one_gain_override_keeps_the_others(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=barkour")
    own = actuators(env_base(cfg.robot).mjx_model)
    cfg.robot.kp = 50.0
    kp, bias_kp, bias_kd, damping = actuators(env_base(cfg.robot).mjx_model)
    np.testing.assert_allclose(kp, 50.0)
    np.testing.assert_allclose(bias_kp, -50.0)
    np.testing.assert_array_equal(bias_kd, own[2])
    np.testing.assert_array_equal(damping, own[3])


def test_null_gains_keep_the_robots_own_model() -> None:
    robot = default_robot(Robot.barkour)
    kp, _, bias_kd, damping = actuators(env_base(robot).mjx_model)
    np.testing.assert_allclose(kp, 35.0)
    np.testing.assert_allclose(bias_kd, -0.5)
    np.testing.assert_allclose(damping, 0.5239)


def test_scale_override_reaches_arena_threshold_and_scan(
    compose_cfg: Compose,
) -> None:
    cfg = compose_cfg(
        "eval",
        "checkpoint=c.pkl",
        "name=t",
        "out_dir=o",
        "robot=go2",
        "robot.scale=1.5",
    )
    config = hfield_config(cfg.robot, ARENA)
    assert config.patch_radius == pytest.approx(1.5 * 1.5)
    assert config.amplitude == pytest.approx(0.3 * 1.5)
    assert config.reach == pytest.approx(2.0 * 1.5)
    assert min_travel(cfg.robot, cfg.min_travel) == pytest.approx(1.5)
    base = env_base(cfg.robot)
    own = env_base(default_robot(Robot.go2))
    assert base.scale == 1.5 and own.scale == ROBOT_SCALE["go2"]
    np.testing.assert_allclose(
        np.asarray(base._scan_grid), np.asarray(own._scan_grid) * 1.5 / 2.5
    )


def test_scale_override_reaches_travel_bounds(compose_cfg: Compose) -> None:
    small = compose_cfg("train", "mode=dr", "name=t", "robot=go2")
    small.robot.scale = 1.25
    config = hfield_config(small.robot, ARENA)
    env = make_env(small.robot, Backend.jax, config, num_envs=1, train=True)
    own = default_robot(Robot.go2)
    own_env = make_env(
        own, Backend.jax, hfield_config(own, ARENA), num_envs=1, train=True
    )
    assert env.travel.sensing == pytest.approx(own_env.travel.sensing / 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("kp", 10.0), ("kd", 1.0), ("damping", 1.0), ("scale", 2.0)],
)
def test_wojtek_refuses_playground_overrides(
    compose_cfg: Compose, field: str, value: float
) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", f"robot.{field}={value}")
    with pytest.raises(ValueError, match=f"robot.{field}|robot.kp/kd/damping"):
        config = hfield_config(cfg.robot, ARENA)
        make_env(cfg.robot, Backend.jax, config, num_envs=1, train=False)


def test_scale_must_be_positive(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=go1")
    cfg.robot.scale = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        robot_scale(cfg.robot)
