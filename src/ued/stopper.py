from collections import deque


class PlateauStopper:
    def __init__(
        self, window: int, patience: int, rel_tol: float, min_iterations: int
    ) -> None:
        self._scores: deque[float] = deque(maxlen=window)
        self._window = window
        self._patience = patience
        self._rel_tol = rel_tol
        self._min_iterations = min_iterations
        self._best = float("inf")
        self._stale = 0

    def update(self, score: float, iteration: int) -> bool:
        self._scores.append(score)
        if iteration < self._min_iterations or len(self._scores) < self._window:
            return False
        mean = sum(self._scores) / len(self._scores)
        if mean < self._best * (1.0 - self._rel_tol):
            self._best = mean
            self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self._patience

    @property
    def smoothed_regret(self) -> float:
        if not self._scores:
            return float("nan")
        return sum(self._scores) / len(self._scores)
