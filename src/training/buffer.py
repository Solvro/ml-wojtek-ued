import numpy as np


class LevelBuffer:
    """Highest-scoring levels, sampled rank-prioritized with staleness."""

    def __init__(
        self, capacity: int, temperature: float = 0.3, staleness: float = 0.3
    ) -> None:
        self.capacity = capacity
        self.temperature = temperature
        self.staleness = staleness
        self.levels: list[np.ndarray] = []
        self.scores = np.zeros(0)
        self.last_seen = np.zeros(0)
        self.roughness = np.zeros(0)
        # (fast, slow) success EMAs per level, the learning-progress state
        self.progress = np.zeros((0, 2))

    def __len__(self) -> int:
        return len(self.levels)

    def add(
        self,
        levels: np.ndarray,
        scores: np.ndarray,
        now: int,
        roughness: np.ndarray,
        progress: np.ndarray,
    ) -> None:
        self.levels.extend(levels)
        self.scores = np.concatenate([self.scores, scores])
        seen = np.full(len(scores), now)
        self.last_seen = np.concatenate([self.last_seen, seen])
        self.roughness = np.concatenate([self.roughness, roughness])
        self.progress = np.concatenate([self.progress, progress])
        if len(self.levels) > self.capacity:
            keep = np.argsort(self.scores)[-self.capacity :]
            self.levels = [self.levels[i] for i in keep]
            self.scores = self.scores[keep]
            self.last_seen = self.last_seen[keep]
            self.roughness = self.roughness[keep]
            self.progress = self.progress[keep]

    def sample(self, rng: np.random.Generator, n: int, now: int) -> np.ndarray:
        ranks = np.empty(len(self)).astype(int)
        ranks[np.argsort(-self.scores)] = np.arange(1, len(self) + 1)
        score_p = (1.0 / ranks) ** (1.0 / self.temperature)
        score_p /= score_p.sum()
        stale = now - self.last_seen + 1e-6
        stale_p = stale / stale.sum()
        p = (1 - self.staleness) * score_p + self.staleness * stale_p
        return rng.choice(len(self), size=n, p=p)

    def update(
        self,
        idx: np.ndarray,
        scores: np.ndarray,
        now: int,
        progress: np.ndarray,
    ) -> None:
        self.scores[idx] = scores
        self.last_seen[idx] = now
        self.progress[idx] = progress

    def stats(self) -> dict[str, float]:
        if len(self) < 2:
            return {}
        rough = self.roughness
        score_ranks = np.argsort(np.argsort(self.scores))
        rough_ranks = np.argsort(np.argsort(rough))
        corr = np.corrcoef(score_ranks, rough_ranks)[0, 1]
        return {
            "buffer_roughness_mean": float(rough.mean()),
            "buffer_roughness_p90": float(np.percentile(rough, 90)),
            "buffer_score_roughness_spearman": float(corr)
            if np.isfinite(corr)
            else 0.0,
        }
