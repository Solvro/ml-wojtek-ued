from collections.abc import Callable

import numpy as np

BOOTSTRAP_SAMPLES = 1000


def cross_fit_cvar(a: np.ndarray, b: np.ndarray, alpha: float) -> float:
    """Worst-alpha levels by one pass, scored on the other, both ways:
    ranking and scoring on the same episodes would bias CVaR low."""
    k = max(1, int(np.ceil(alpha * len(a))))
    return 0.5 * (b[np.argsort(a)[:k]].mean() + a[np.argsort(b)[:k]].mean())


def bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """95% CI of `statistic(indices)` over n resampled levels."""
    samples = [
        statistic(rng.integers(0, n, n)) for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return float(np.percentile(samples, 2.5)), float(
        np.percentile(samples, 97.5)
    )


def envelope(
    values: np.ndarray, success: np.ndarray, threshold: float = 0.5
) -> float:
    """Largest |feature| still passed at `threshold` success, interpolated."""
    if success[0] < threshold:
        return 0.0
    for i in range(1, len(values)):
        if success[i] < threshold:
            frac = (success[i - 1] - threshold) / (success[i - 1] - success[i])
            return float(values[i - 1] + frac * (values[i] - values[i - 1]))
    return float(values[-1])
