import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from config.schema import EvalConfig, Sim2SimConfig, TrainConfig
from robots.registry import TERRAIN_ROBOTS
from robots.scale import level_steps, min_travel, robot_scale


@dataclasses.dataclass(frozen=True)
class RobotRecord:
    """The robot a checkpoint was trained as, written beside it. Gains are
    read off the compiled model; None for Wojtek (wojtek_env sets them)."""

    name: str
    kp: float | None
    kd: float | None
    damping: float | None
    scale: float
    min_travel_m: float
    level_steps: int


def record_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".robot.yaml")


def uniform(values: Any, what: str) -> float:
    values = np.asarray(values)
    if not np.all(values == values[0]):
        raise ValueError(f"{what} differs between joints: {values.tolist()}")
    return float(values[0])


def robot_record(
    cfg: TrainConfig | EvalConfig | Sim2SimConfig, base: Any
) -> RobotRecord:
    """`base`: the robot env `make_env` built for `cfg.robot`."""
    robot = cfg.robot
    gains: tuple[float | None, ...] = (None, None, None)
    if robot.name in TERRAIN_ROBOTS:
        model = base.mj_model
        gains = (
            uniform(model.actuator_gainprm[:, 0], "actuator kp"),
            # 0.0 - : a zero kv (Go1) records as 0.0, not -0.0
            uniform(0.0 - model.actuator_biasprm[:, 2], "actuator kv"),
            uniform(model.dof_damping[6:], "leg joint damping"),
        )
    return RobotRecord(
        str(robot.name),
        *gains,
        scale=robot_scale(robot),
        min_travel_m=min_travel(robot, cfg.min_travel),
        level_steps=level_steps(robot, cfg.level_steps),
    )


def write_record(checkpoint: Path, record: RobotRecord) -> None:
    OmegaConf.save(
        OmegaConf.create(dataclasses.asdict(record)), record_path(checkpoint)
    )


def read_record(checkpoint: Path) -> RobotRecord | None:
    """None when the run left no record beside its checkpoint."""
    path = record_path(checkpoint)
    if not path.is_file():
        return None
    fields: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(fields, dict):
        raise TypeError(f"{path} is not a mapping of robot settings")
    return RobotRecord(**fields)


def same(a: float | str | None, b: float | str | None) -> bool:
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    return math.isclose(a, b, rel_tol=1e-6)


def check_record(checkpoint: Path, record: RobotRecord) -> None:
    """Refuses an eval whose robot differs from the checkpoint's record."""
    trained = read_record(checkpoint)
    if trained is None:
        warnings.warn(
            f"no {record_path(checkpoint)}: cannot check that "
            f"robot={record.name} and its robot.* overrides match the "
            "checkpoint's training",
            stacklevel=2,
        )
        return
    diff = {
        f: (getattr(trained, f), getattr(record, f))
        for f in (f.name for f in dataclasses.fields(RobotRecord))
        if not same(getattr(trained, f), getattr(record, f))
    }
    if diff:
        lines = ", ".join(
            f"{f} trained {t} vs now {n}" for f, (t, n) in diff.items()
        )
        raise ValueError(
            f"{checkpoint} was trained with a different robot ({lines}); "
            f"see {record_path(checkpoint)} and pass the run's robot.* "
            "overrides (ROBOT_ARGS in slurm/), min_travel and level_steps"
        )
