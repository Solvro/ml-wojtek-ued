from __future__ import annotations

import numpy as np

from config.hfield_config import HFieldConfig
from wojtek_env.terrain.terrain import _bilinear


class HFieldManager:
    def __init__(self, config: HFieldConfig | None = None):
        self.config = config or HFieldConfig()

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        n = self.config.coarse_n
        return rng.uniform(0.0, 1.0, size=(n, n))

    def mutate(self, level: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.config
        out = level.copy()
        rows = rng.integers(0, c.coarse_n, size=c.n_edits)
        cols = rng.integers(0, c.coarse_n, size=c.n_edits)
        nudge = rng.uniform(-c.edit_scale, c.edit_scale, size=c.n_edits)
        out[rows, cols] = np.clip(out[rows, cols] + nudge, 0.0, 1.0)
        return out

    def to_hfield_data(self, level: np.ndarray) -> np.ndarray:
        c = self.config
        half = c.patch_radius
        step = 2 * half / (c.coarse_n - 1)
        xs = np.linspace(-half, half, c.grid_size)
        ys = np.linspace(-half, half, c.grid_size)
        lx, ly = np.meshgrid(xs, ys)
        fine = _bilinear(level, -half, step, -half, step, lx, ly)
        taper = self._pad_taper(lx, ly)
        return (0.5 + (fine - 0.5) * taper * c.difficulty).astype(np.float32)

    def height_range(self, data: np.ndarray) -> tuple[float, float]:
        a = self.config.amplitude
        return 2 * a * float(data.min()) - a, 2 * a * float(data.max()) - a

    def _pad_taper(self, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
        r = np.sqrt(lx**2 + ly**2)
        c = self.config
        return np.clip((r - c.pad_radius) / c.pad_taper, 0.0, 1.0)
