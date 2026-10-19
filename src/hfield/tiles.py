"""The shared tiled arena: level i on tile i of one static heightfield
(docs/environment.md#tiled-arena)."""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from config.hfield_config import HFieldConfig


@dataclass(frozen=True)
class TileLayout:
    config: HFieldConfig

    @property
    def per_side(self) -> int:
        return self.config.tiles

    @property
    def n_tiles(self) -> int:
        return self.config.tiles**2

    @property
    def gap_cells(self) -> int:
        c = self.config
        # even, so each tile gets the same half-gap; at least one flat node
        # on each side keeps neighbouring patches from sharing a node
        half = math.ceil(round((c.reach - c.patch_radius) / c.cell_size, 6) / 2)
        return 2 * max(half, 1)

    @property
    def gap(self) -> float:
        return self.gap_cells * self.config.cell_size

    @property
    def pitch_cells(self) -> int:
        return self.config.grid_size - 1 + self.gap_cells

    @property
    def pitch(self) -> float:
        return self.pitch_cells * self.config.cell_size

    @property
    def nodes(self) -> int:
        """hfield rows (= columns) of the whole arena."""
        return self.per_side * self.pitch_cells + 1

    @property
    def half(self) -> float:
        """Half-width of the arena, m."""
        return self.per_side * self.pitch / 2

    @property
    def leak_radius(self) -> float:
        """Chebyshev distance from a tile centre where the next tile begins."""
        return self.config.patch_radius + self.gap

    def centres(self) -> np.ndarray:
        """(n_tiles, 2) world xy of every tile centre, tile i = level i."""
        index = np.arange(self.n_tiles)
        row, col = index // self.per_side, index % self.per_side
        xy = np.stack([col, row], axis=-1) + 0.5
        return (xy * self.pitch - self.half).astype(np.float32)

    def assemble(self, levels: jax.Array) -> jax.Array:
        """`hfield_data` of the arena: level i (rendered rows, (k, grid**2),
        k <= n_tiles) on tile i, every other node flat."""
        g, n, p = self.config.grid_size, self.per_side, self.pitch_cells
        k = levels.shape[0]
        if k > self.n_tiles:
            raise ValueError(
                f"{k} levels but the arena has {self.n_tiles} tiles "
                f"({n} per side); rebuild it with arena.tiles="
                f"{math.ceil(math.sqrt(k))} or run fewer levels at once"
            )
        flat = jnp.full((self.n_tiles, g, g), 0.5, jnp.float32)
        tiles = flat.at[:k].set(jnp.reshape(levels, (k, g, g)))
        pad = self.gap_cells // 2
        cells = jnp.pad(
            tiles, ((0, 0), (pad, pad), (pad, pad)), constant_values=0.5
        )
        # a cell spans p + 1 nodes; its last row/column is the next cell's first
        cells = cells[:, :p, :p].reshape(n, n, p, p).transpose(0, 2, 1, 3)
        grid = jnp.pad(
            cells.reshape(n * p, n * p), ((0, 1), (0, 1)), constant_values=0.5
        )
        return grid.reshape(-1)

    def tile_xy(self, levels: int, envs_per_level: int) -> jax.Array:
        """(levels * envs_per_level, 2) spawn tile centres, level-major."""
        centres = jnp.asarray(self.centres()[:levels])
        return jnp.repeat(centres, envs_per_level, axis=0)

    def leaked(
        self, xy: jax.Array, tile_xy: jax.Array, sensing: float = 0.0
    ) -> jax.Array:
        """Whether a robot at `xy` reads terrain past its own tile's gap, from
        up to `sensing` m away (height scan, feet)."""
        offset = jnp.max(jnp.abs(xy - tile_xy), axis=-1)
        return offset + sensing > self.leak_radius

    def check_reach(self, reach: float, what: str) -> None:
        """Fail when an env can read terrain `reach` m (Chebyshev) from its
        tile centre, past the gap, on a neighbouring lane's level."""
        if reach > self.leak_radius:
            raise ValueError(
                f"{what}: a robot can read terrain {reach:.2f} m from its tile "
                f"centre, past this arena's {self.leak_radius:.2f} m, on a "
                "neighbouring lane's level. Run fewer steps per level, or set "
                f"HFieldConfig.reach >= {math.ceil(reach)} and rebuild the "
                "arena"
            )
