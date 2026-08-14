from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import jax
import numpy as np

from hfield.manager import HFieldManager
from modules.ppo_modules import A2C, Episode
from utils.buffer import LevelBuffer


class RolloutFn(Protocol):
    def __call__(
        self, key: jax.Array, model: A2C, *, hfield_data: jax.Array | None = None
    ) -> Episode: ...


class Curator(NamedTuple):
    buffer: LevelBuffer
    hfield_manager: HFieldManager
    rng: np.random.Generator
    replay_prob: float
    regret_threshold: float


class BatchCurator(NamedTuple):
    buffer: LevelBuffer
    hfield_manager: HFieldManager
    rng: np.random.Generator
    replay_prob: float
    num_lanes: int
    top_k: int


class PPOTrainer(NamedTuple):
    rollout: RolloutFn
    advantage: Callable[[A2C, Episode], tuple[jax.Array, jax.Array]]
    make_step: Callable[..., tuple[A2C, Any, jax.Array]]
    epochs: int
    minibatches: int
