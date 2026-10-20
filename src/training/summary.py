import dataclasses
from collections import deque
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

FINAL_WINDOW = 10  # iterations averaged into the final training metrics
METRICS = (
    "reward_per_step",
    "train_success",
    "falls_per_step",
    "forward_vel",
    "cmd_vx",
)


@dataclasses.dataclass(frozen=True)
class TrainingSummary:
    """Training-time metrics of a run: means over its last iterations."""

    env_steps: int
    wall_time_s: float
    window_iterations: int
    reward_per_step: float
    train_success: float
    falls_per_step: float
    forward_vel: float
    cmd_vx: float


def summary_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".train.yaml")


class SummaryWindow:
    """The last `FINAL_WINDOW` iterations' training metrics."""

    def __init__(self) -> None:
        self.rows: deque[dict[str, float]] = deque(maxlen=FINAL_WINDOW)

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append({k: float(row[k]) for k in METRICS})

    def summary(
        self, env_steps: int, wall_time_s: float
    ) -> TrainingSummary | None:
        n = len(self.rows)
        if not n:
            return None
        means = {k: sum(r[k] for r in self.rows) / n for k in METRICS}
        return TrainingSummary(
            env_steps=env_steps,
            wall_time_s=wall_time_s,
            window_iterations=n,
            **means,
        )


def write_summary(checkpoint: Path, summary: TrainingSummary) -> None:
    OmegaConf.save(
        OmegaConf.create(dataclasses.asdict(summary)), summary_path(checkpoint)
    )


def read_summary(checkpoint: Path) -> TrainingSummary | None:
    """None when the run left no summary beside its checkpoint."""
    path = summary_path(checkpoint)
    if not path.is_file():
        return None
    fields: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(fields, dict):
        raise TypeError(f"{path} is not a mapping of training metrics")
    return TrainingSummary(**fields)
