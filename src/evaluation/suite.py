from collections.abc import Iterator

import numpy as np

from config.hfield_config import HFieldConfig

SUITE_TERRAINS = (
    "stairs_up",
    "stairs_down",
    "slope_up",
    "slope_down",
    "trenches",
    "waves",
    "blocks",
)
SUITE_STRENGTHS = (1, 2, 3, 4)


def suite_heights(
    terrain: str,
    k: int,
    d: np.ndarray,
    xy: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Height (m) at strength k; d = radial distance past the spawn pad.
    Radial, so the spawn yaw and command heading do not matter."""
    if terrain in ("stairs_up", "stairs_down"):
        h = 0.02 * k * np.ceil(d / 0.3)
        return h if terrain == "stairs_up" else -h
    if terrain in ("slope_up", "slope_down"):
        h = np.tan(np.radians(5.0 * k)) * d
        return h if terrain == "slope_up" else -h
    if terrain == "trenches":
        return np.where((d % 0.5) < 0.05 * k, -0.1, 0.0)
    if terrain == "waves":
        return 0.02 * k * np.sin(2 * np.pi * d / 0.6)
    if terrain == "blocks":
        cells = np.floor(xy / 0.3).astype(int)
        cells -= cells.min(axis=(0, 1))
        heights = rng.uniform(-0.02 * k, 0.02 * k, cells.max(axis=(0, 1)) + 1)
        return heights[cells[..., 0], cells[..., 1]]
    raise ValueError(f"unknown suite terrain {terrain!r}")


def suite_hfield(
    config: HFieldConfig,
    terrain: str,
    k: int,
    rng: np.random.Generator,
    robot_scale: float,
) -> np.ndarray:
    """A held-out suite level, rendered as one tile's hfield rows."""
    half = config.patch_radius
    axis = np.linspace(-half, half, config.grid_size)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x**2 + y**2)
    start = config.pad_radius + config.pad_taper
    taper = np.clip((r - config.pad_radius) / config.pad_taper, 0.0, 1.0)
    heights = np.zeros_like(r)
    outside = r >= start
    xy = np.stack([x, y], axis=-1)
    d = np.maximum(r - start, 0.0)
    # geometric scaling: same slopes, every length x robot_scale
    s = robot_scale
    suite = suite_heights(terrain, k, d / s, xy / s, rng)
    heights[outside] = s * suite[outside]
    data = 0.5 + heights * taper / (2 * config.amplitude)
    return np.clip(data, 0.0, 1.0).astype(np.float32).reshape(-1)


def suite_levels(
    config: HFieldConfig,
    strengths: tuple[int, ...],
    instances: int,
    rng: np.random.Generator,
    robot_scale: float,
) -> Iterator[tuple[str, np.ndarray]]:
    """(`suite/<terrain>_<k>`, (instances, N) hfields) per cell, in
    eval_brax's order; only blocks draw from `rng`."""
    for terrain in SUITE_TERRAINS:
        for k in strengths:
            hfields = [
                suite_hfield(config, terrain, k, rng, robot_scale)
                for _ in range(instances)
            ]
            yield f"suite/{terrain}_{k}", np.stack(hfields)
