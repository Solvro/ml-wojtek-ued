import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from config.schema import TrainConfig
from env.functional import FunctionalEnv
from hfield.level_space import LevelSpace
from hfield.tiles import TileLayout
from training.adversary import Adversary
from training.student import Student
from training.terrain_cl import TerrainLanes
from training.unroll import Unroll


class KeyStream:
    """A fresh jax key per call, split off one seeded key."""

    def __init__(self, seed: int) -> None:
        self._key = jax.random.key(seed)

    def __call__(self) -> jax.Array:
        self._key, key = jax.random.split(self._key)
        return key


@dataclasses.dataclass
class TrainContext:
    cfg: TrainConfig
    env: FunctionalEnv
    student: Student
    # PAIRED's second student and learned level designer, None elsewhere
    antagonist: Student | None
    adversary: Adversary | None
    space: LevelSpace
    layout: TileLayout
    assemble: Callable[[jax.Array], jax.Array]
    rng: np.random.Generator
    next_key: KeyStream
    reset: Callable[..., Any]
    eval_reset: Callable[..., Any]
    unroll: Unroll
    eval_unroll: Unroll
    # ACCEL mutations queued for the next explore iteration, with the
    # learning-progress state they inherit from their parents
    children: np.ndarray
    children_progress: np.ndarray
    # Terrain-CL's lanes, None elsewhere
    terrain: TerrainLanes | None
