import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from config.hfield_config import HFieldConfig
from config.schema import Backend
from env.factory import make_env
from robots.record import (
    RobotRecord,
    check_record,
    read_record,
    record_path,
    robot_record,
    write_record,
)
from robots.scale import hfield_config

Compose = Callable[..., Any]
ARENA = HFieldConfig(
    patch_radius=1.5,
    amplitude=0.3,
    pad_radius=0.55,
    pad_taper=0.2,
    tiles=2,
    reach=2.0,
)
GO2 = RobotRecord("go2", 100.0, 0.707, 2.828, 2.7, 2.7, 300)


def env_base(robot: Any) -> Any:
    config = hfield_config(robot, ARENA)
    env = make_env(robot, Backend.jax, config, num_envs=1, train=False)
    return env._env.unwrapped


def test_record_sits_beside_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run_s0.pkl"
    assert record_path(checkpoint) == tmp_path / "run_s0.robot.yaml"


def test_record_roundtrips(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c.pkl"
    write_record(checkpoint, GO2)
    assert read_record(checkpoint) == GO2


def test_missing_record_reads_as_none(tmp_path: Path) -> None:
    assert read_record(tmp_path / "c.pkl") is None


def test_malformed_record_raises(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c.pkl"
    record_path(checkpoint).write_text("- not\n- a mapping\n")
    with pytest.raises(TypeError, match="not a mapping"):
        read_record(checkpoint)


def test_matching_robot_passes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c.pkl"
    write_record(checkpoint, GO2)
    check_record(checkpoint, dataclasses.replace(GO2, kd=0.7070000001))


def test_mismatched_robot_names_the_field(tmp_path: Path) -> None:
    checkpoint = tmp_path / "c.pkl"
    write_record(checkpoint, GO2)
    now = dataclasses.replace(GO2, kp=50.0, scale=2.5, level_steps=600)
    with pytest.raises(ValueError, match="kp trained 100.0 vs now 50.0") as e:
        check_record(checkpoint, now)
    assert "scale trained 2.7 vs now 2.5" in str(e.value)
    assert "level_steps trained 300 vs now 600" in str(e.value)
    assert "ROBOT_ARGS" in str(e.value)


def test_missing_record_warns(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="cannot check that robot=go2"):
        check_record(tmp_path / "c.pkl", GO2)


def test_wojtek_record_has_no_gains(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t")
    record = robot_record(cfg, None)
    assert record == RobotRecord("wojtek", None, None, None, 1.0, 1.0, 300)


def test_record_reads_the_compiled_gains(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=go2")
    record = robot_record(cfg, env_base(cfg.robot))
    assert record.name == "go2"
    assert record.kp == pytest.approx(100.0)
    assert record.kd == pytest.approx(0.707)
    assert record.damping == pytest.approx(2.828)
    assert (record.scale, record.min_travel_m) == pytest.approx((2.7, 2.7))
    cfg.robot.kp = 70.0
    assert robot_record(cfg, env_base(cfg.robot)).kp == 70.0


def test_spot_record_carries_its_level_length(compose_cfg: Compose) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=t", "robot=spot")
    record = robot_record(cfg, env_base(cfg.robot))
    assert (record.scale, record.min_travel_m) == pytest.approx((4.3, 4.3))
    assert record.level_steps == 780
