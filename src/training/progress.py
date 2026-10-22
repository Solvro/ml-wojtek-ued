import numpy as np

from config.schema import ModeConfig


def update_progress(
    prior: np.ndarray, success: np.ndarray, mode: ModeConfig
) -> np.ndarray:
    """Per level (fast, slow) EMAs of the lane success fraction after one
    more visit; a NaN row (first visit) starts both at its success."""
    rates = np.array([mode.lp_fast_rate, mode.lp_slow_rate])
    p = success[:, None]
    return np.where(np.isnan(prior), p, prior + rates * (p - prior))


def learning_progress(progress: np.ndarray) -> np.ndarray:
    return np.abs(progress[:, 0] - progress[:, 1])


def unvisited(n: int) -> np.ndarray:
    return np.full((n, 2), np.nan)
