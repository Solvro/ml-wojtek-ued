import jax
import jax.numpy as jnp
import numpy as np

from hfield.manager import HFieldManager
from utils.buffer import LevelBuffer


def level_to_hfield_data(hfield_manager: HFieldManager, level: np.ndarray) -> jax.Array:
    return jnp.asarray(hfield_manager.to_hfield_data(level).reshape(-1))


def level_to_hfield_data_batch(
    hfield_manager: HFieldManager, levels: np.ndarray
) -> jax.Array:
    return jnp.stack([level_to_hfield_data(hfield_manager, level) for level in levels])


def sample_from_buffer(buffer: LevelBuffer, rng: np.random.Generator) -> np.ndarray:
    levels = buffer.get()
    return levels[rng.integers(len(levels))]


def sample_batch_from_buffer(
    buffer: LevelBuffer, rng: np.random.Generator, num_lanes: int
) -> np.ndarray:
    levels = buffer.get()
    idx = rng.integers(len(levels), size=num_lanes)
    return np.stack([levels[i] for i in idx])


def generate_batch(
    hfield_manager: HFieldManager, rng: np.random.Generator, num_lanes: int
) -> np.ndarray:
    return np.stack([hfield_manager.generate(rng) for _ in range(num_lanes)])


def mutate_batch(
    hfield_manager: HFieldManager, levels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return np.stack([hfield_manager.mutate(level, rng) for level in levels])


def fill_buffer(
    buffer: LevelBuffer,
    hfield_manager: HFieldManager,
    rng: np.random.Generator,
    fill_ratio: float,
) -> None:
    for _ in range(int(buffer.size * fill_ratio)):
        buffer.add(hfield_manager.generate(rng))
