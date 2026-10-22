import numpy as np


def learning_progress(prior: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """LP-ACRL's learning progress per level, signed: this visit's mean
    episodic return minus the previous visit's (`prior`); a NaN prior (a
    fresh level's first visit) scores 0."""
    return np.where(np.isnan(prior), 0.0, returns - prior)


def unvisited(n: int) -> np.ndarray:
    return np.full(n, np.nan)
