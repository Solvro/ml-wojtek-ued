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
    level_steps,
    min_travel,
    pd_gains,
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


@pytest.mark.parametrize("robot", ["go1", "go2", "barkour", "spot"])
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
    cfg.robot.kp = 70.0
    kp, bias_kp, bias_kd, damping = actuators(env_base(cfg.robot).mjx_model)
    np.testing.assert_allclose(kp, 70.0)
    np.testing.assert_allclose(bias_kp, -70.0)
    np.testing.assert_array_equal(bias_kd, own[2])
    np.testing.assert_array_equal(damping, own[3])


# (kp, kv, joint damping) each robot compiles to with no robot.* override
OWN_GAINS = {
    "go1": (35.0, 0.0, 0.5),
    "go2": (100.0, 0.707, 2.828),
    "barkour": (50.0, 0.5, 0.024),
    "spot": (300.0, 20.0, 1.0),
}


@pytest.mark.parametrize("robot", list(OWN_GAINS))
def test_null_gains_compile_the_robots_own_gains(robot: str) -> None:
    kp_own, kd_own, damping_own = OWN_GAINS[robot]
    base = env_base(default_robot(Robot(robot)))
    for model in (base.mj_model, base.mjx_model):
        kp, bias_kp, bias_kd, damping = actuators(model)
        np.testing.assert_allclose(kp, kp_own, rtol=1e-6)
        np.testing.assert_allclose(bias_kp, -kp_own, rtol=1e-6)
        np.testing.assert_allclose(bias_kd, -kd_own, rtol=1e-6)
        np.testing.assert_allclose(damping, damping_own, rtol=1e-6)


@pytest.mark.parametrize(
    ("robot", "scale", "steps"),
    [
        ("wojtek", 1.0, 300),
        ("go1", 2.5, 300),
        ("go2", 2.7, 300),
        ("barkour", 2.5, 300),
        ("spot", 4.3, 780),
    ],
)
def test_own_scale_threshold_and_level_length(
    robot: str, scale: float, steps: int
) -> None:
    own = default_robot(Robot(robot))
    assert robot_scale(own) == pytest.approx(scale)
    assert min_travel(own, 1.0) == pytest.approx(scale)
    assert level_steps(own, 300) == steps


def test_scale_override_keeps_the_level_length(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=spot")
    cfg.robot.scale = 2.0
    assert level_steps(cfg.robot, cfg.level_steps) == 780


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
        np.asarray(base._scan_grid),
        np.asarray(own._scan_grid) * 1.5 / ROBOT_SCALE["go2"],
    )


def test_scale_override_reaches_travel_bounds(compose_cfg: Compose) -> None:
    small = compose_cfg("train", "mode=dr", "name=t", "robot=go2")
    small.robot.scale = ROBOT_SCALE["go2"] / 2
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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kp", 0.0, "robot.kp=0.0 must be > 0"),
        ("kd", -0.1, "robot.kd=-0.1 must be >= 0"),
        ("damping", -1.0, "robot.damping=-1.0 must be >= 0"),
    ],
)
def test_gains_must_be_physical(
    compose_cfg: Compose, field: str, value: float, match: str
) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=go2")
    setattr(cfg.robot, field, value)
    with pytest.raises(ValueError, match=match):
        pd_gains(cfg.robot)


def test_zero_kd_and_damping_are_allowed(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=go2")
    cfg.robot.kd = 0.0
    cfg.robot.damping = 0.0
    gains = pd_gains(cfg.robot)
    assert (gains.kd, gains.damping) == (0.0, 0.0)
