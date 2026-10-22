"""Level spaces (docs/levels.md#level-spaces): `flat` is a featureless
genome, `shrink(levels, d)` scales each level toward it by its d (0 = flat),
`scaled(d)` shrinks a uniform draw."""

from typing import NamedTuple, Protocol

import numpy as np

from config.hfield_config import HFieldConfig
from hfield.manager import HFieldManager
from wojtek_env.terrain.terrain import _bilinear


class LevelSpace(Protocol):
    shape: tuple[int, ...]
    gen_shape: tuple[int, ...]
    config: HFieldConfig

    def uniform(self, rng: np.random.Generator, n: int) -> np.ndarray: ...
    def shrink(
        self, levels: np.ndarray, difficulty: np.ndarray
    ) -> np.ndarray: ...
    def scaled(
        self, rng: np.random.Generator, n: int, difficulty: float
    ) -> np.ndarray: ...
    def easy(
        self, rng: np.random.Generator, n: int, scale: float
    ) -> np.ndarray: ...
    def mutate(
        self,
        level: np.ndarray,
        rng: np.random.Generator,
        grow_max: float | None,
    ) -> np.ndarray: ...
    def from_generated(
        self, theta: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray: ...
    def render(self, levels: np.ndarray) -> np.ndarray: ...


def mean_abs_height(config: HFieldConfig, hfield: np.ndarray) -> np.ndarray:
    """Mean |height| (m) past the spawn pad, per rendered level (rows)."""
    half = config.patch_radius
    axis = np.linspace(-half, half, config.grid_size)
    x, y = np.meshgrid(axis, axis)
    outside = (
        np.sqrt(x**2 + y**2) >= config.pad_radius + config.pad_taper
    ).reshape(-1)
    heights = (np.asarray(hfield)[:, outside] - 0.5) * 2 * config.amplitude
    return np.abs(heights).mean(axis=1)


class GridSpace:
    def __init__(self, config: HFieldConfig, n_edits: int, edit_scale: float):
        self.config = config
        self.n_edits = n_edits
        self.edit_scale = edit_scale
        self.manager = HFieldManager(config)
        self.shape: tuple[int, ...] = (config.coarse_n, config.coarse_n)
        self.gen_shape: tuple[int, ...] = self.shape

    def uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(0.0, 1.0, (n, *self.shape))

    def shrink(self, levels: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
        return 0.5 + (levels - 0.5) * difficulty[:, None, None]

    def scaled(
        self, rng: np.random.Generator, n: int, difficulty: float
    ) -> np.ndarray:
        return self.shrink(self.uniform(rng, n), np.full(n, difficulty))

    def easy(
        self, rng: np.random.Generator, n: int, scale: float
    ) -> np.ndarray:
        return np.clip(
            0.5 + rng.uniform(-scale, scale, (n, *self.shape)), 0.0, 1.0
        )

    def mutate(
        self,
        level: np.ndarray,
        rng: np.random.Generator,
        grow_max: float | None,
    ) -> np.ndarray:
        child = self.manager.mutate(level, rng, self.n_edits, self.edit_scale)
        if grow_max:
            child = 0.5 + (child - 0.5) * rng.uniform(1.0, grow_max)
        return np.clip(child, 0.0, 1.0)

    def from_generated(
        self, theta: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return np.asarray(theta)

    def render(self, levels: np.ndarray) -> np.ndarray:
        return np.stack(
            [self.manager.to_hfield_data(level).reshape(-1) for level in levels]
        )


class Param(NamedTuple):
    name: str
    low: float
    high: float
    flat: float  # genome value of "no feature"


# ranges for a 0.10 m robot: uphill, step-up, gap and block maxima sit at the
# 50%-success envelope of policies trained on a wider box (docs/levels.md)
PARAMS = (
    Param("rough_amp_m", 0.0, 0.10, 0.0),
    Param("rough_wavelength_m", 0.2, 1.0, 0.5),
    Param("slope_deg", -20.0, 18.0, 20.0 / 38.0),
    Param("step_height_m", -0.08, 0.05, 0.08 / 0.13),
    Param("step_width_m", 0.15, 0.60, 0.5),
    Param("gap_width_m", 0.0, 0.06, 0.0),
    Param("block_height_m", 0.0, 0.06, 0.0),
    Param("block_size_m", 0.15, 0.60, 0.5),
)
GAP_DEPTH_M = 0.15
GAP_PERIOD_M = 0.6


class ParamSpace:
    """8 terrain parameters in [0, 1] plus a never-mutated seed slot last."""

    def __init__(
        self,
        config: HFieldConfig,
        n_edits: int = 2,
        edit_scale: float = 0.1,
        robot_scale: float = 1.0,
    ):
        self.config = config
        # robot size relative to Wojtek: every length range (not slope) scales
        self.robot_scale = robot_scale
        self.n_edits = n_edits
        self.edit_scale = edit_scale
        self.gen_shape: tuple[int, ...] = (len(PARAMS),)
        self.shape: tuple[int, ...] = (len(PARAMS) + 1,)
        self.flat = np.array([p.flat for p in PARAMS])
        half = config.patch_radius
        axis = np.linspace(-half, half, config.grid_size)
        self.x, self.y = np.meshgrid(axis, axis)
        r = np.sqrt(self.x**2 + self.y**2)
        start = config.pad_radius + config.pad_taper
        self.d = np.maximum(r - start, 0.0)
        self.taper = np.clip(
            (r - config.pad_radius) / config.pad_taper, 0.0, 1.0
        )

    def _with_seed(
        self, genome: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        seeds = rng.uniform(0.0, 1.0, (len(genome), 1))
        return np.concatenate([genome, seeds], axis=1)

    def uniform(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self._with_seed(rng.uniform(0.0, 1.0, (n, len(PARAMS))), rng)

    def shrink(self, levels: np.ndarray, difficulty: np.ndarray) -> np.ndarray:
        genome = levels[:, : len(PARAMS)]
        shrunk = self.flat + (genome - self.flat) * difficulty[:, None]
        return np.concatenate([shrunk, levels[:, len(PARAMS) :]], axis=1)

    def scaled(
        self, rng: np.random.Generator, n: int, difficulty: float
    ) -> np.ndarray:
        return self.shrink(self.uniform(rng, n), np.full(n, difficulty))

    def easy(
        self, rng: np.random.Generator, n: int, scale: float
    ) -> np.ndarray:
        return self.scaled(rng, n, scale)

    def mutate(
        self,
        level: np.ndarray,
        rng: np.random.Generator,
        grow_max: float | None,
    ) -> np.ndarray:
        child = level.copy()
        idx = rng.choice(
            len(PARAMS), size=min(self.n_edits, len(PARAMS)), replace=False
        )
        child[idx] += rng.uniform(-self.edit_scale, self.edit_scale, len(idx))
        if grow_max:
            genome = child[: len(PARAMS)]
            child[: len(PARAMS)] = self.flat + (
                genome - self.flat
            ) * rng.uniform(1.0, grow_max)
        child[: len(PARAMS)] = np.clip(child[: len(PARAMS)], 0.0, 1.0)
        return child

    def from_generated(
        self, theta: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return self._with_seed(np.asarray(theta), rng)

    def physical(self, level: np.ndarray) -> dict[str, float]:
        return {
            p.name: (p.low + float(level[i]) * (p.high - p.low))
            * (self.robot_scale if p.name.endswith("_m") else 1.0)
            for i, p in enumerate(PARAMS)
        }

    def heights(self, level: np.ndarray) -> np.ndarray:
        p = self.physical(level)
        rng = np.random.default_rng(int(level[-1] * 2**31))
        d = self.d
        h = np.tan(np.radians(p["slope_deg"])) * d
        h = h + p["step_height_m"] * np.ceil(d / p["step_width_m"])
        h = h + p["rough_amp_m"] * self._field(
            rng, p["rough_wavelength_m"], smooth=True
        )
        h = h + p["block_height_m"] * self._field(
            rng, p["block_size_m"], smooth=False
        )
        period, depth = (
            GAP_PERIOD_M * self.robot_scale,
            GAP_DEPTH_M * self.robot_scale,
        )
        gaps = (d > 0) & ((d % period) < p["gap_width_m"])
        h = np.where(gaps, h - depth, h)
        return h * self.taper

    def _field(
        self, rng: np.random.Generator, cell: float, smooth: bool
    ) -> np.ndarray:
        """Seeded field in [-1, 1]: bilinear noise, or constant blocks."""
        half = self.config.patch_radius
        n = int(np.ceil(2 * half / cell)) + 1
        values = rng.uniform(-1.0, 1.0, (n, n))
        if smooth:
            return _bilinear(values, -half, cell, -half, cell, self.x, self.y)
        cols = np.floor((self.x + half) / cell).astype(int)
        rows = np.floor((self.y + half) / cell).astype(int)
        return values[np.minimum(rows, n - 1), np.minimum(cols, n - 1)]

    def render(self, levels: np.ndarray) -> np.ndarray:
        amplitude = self.config.amplitude
        return np.stack(
            [
                np.clip(0.5 + self.heights(level) / (2 * amplitude), 0.0, 1.0)
                .astype(np.float32)
                .reshape(-1)
                for level in levels
            ]
        )


def make_level_space(
    kind: str,
    config: HFieldConfig,
    n_edits: int,
    edit_scale: float,
    robot_scale: float = 1.0,
) -> LevelSpace:
    if kind == "grid":
        return GridSpace(config, n_edits=n_edits, edit_scale=edit_scale)
    if kind == "param":
        return ParamSpace(
            config,
            n_edits=n_edits,
            edit_scale=edit_scale,
            robot_scale=robot_scale,
        )
    raise ValueError(f"unknown level space {kind!r}; use 'grid' or 'param'")
