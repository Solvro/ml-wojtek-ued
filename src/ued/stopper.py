import math
from collections import deque


class PlateauStopper:
    def __init__(
        self, window: int, patience: int, rel_tol: float, min_iterations: int
    ) -> None:
        self._regret_scores: deque[float] = deque(maxlen=window)
        self._rewards: deque[float] = deque(maxlen=window)
        self._window = window
        self._patience = patience
        self._rel_tol = rel_tol
        self._min_iterations = min_iterations
        self._best_regret = float("inf")
        self._best_reward = float("-inf")
        self._stale = 0

    def update(self, regret_score: float, mean_reward: float, iteration: int) -> bool:
        self._regret_scores.append(regret_score)
        self._rewards.append(mean_reward)
        if iteration < self._min_iterations or len(self._regret_scores) < self._window:
            return False

        regret_mean = sum(self._regret_scores) / len(self._regret_scores)
        reward_mean = sum(self._rewards) / len(self._rewards)

        regret_improved = regret_mean < self._best_regret * (1.0 - self._rel_tol)
        reward_scale = (
            max(1.0, abs(self._best_reward))
            if math.isfinite(self._best_reward)
            else 1.0
        )
        reward_improved = reward_mean > self._best_reward + self._rel_tol * reward_scale

        if regret_improved:
            self._best_regret = regret_mean
        if reward_improved:
            self._best_reward = reward_mean

        if regret_improved or reward_improved:
            self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self._patience

    @property
    def smoothed_regret(self) -> float:
        if not self._regret_scores:
            return float("nan")
        return sum(self._regret_scores) / len(self._regret_scores)

    @property
    def smoothed_reward(self) -> float:
        if not self._rewards:
            return float("nan")
        return sum(self._rewards) / len(self._rewards)
