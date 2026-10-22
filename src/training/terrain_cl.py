"""Terrain-CL: the game-inspired terrain curriculum of Rudin et al. 2022
(legged_gym `terrain_levels`), one lane = one column of fixed terrain."""

from typing import NamedTuple

import numpy as np

from hfield.level_space import LevelSpace

TOP_ROW = 10  # rows 0..TOP_ROW; row r runs the column shrunk by r / TOP_ROW
MAX_INIT_ROW = 5  # legged_gym max_init_terrain_level
PROMOTE_SUCCESS = 0.5
# legged_gym demotes below half the commanded distance
DEMOTE_TRAVEL = 0.5


class LaneOutcome(NamedTuple):
    success: np.ndarray  # fraction of the lane's envs that succeeded
    travel: np.ndarray  # mean distance from spawn (m)
    cmd_distance: np.ndarray  # mean commanded planar distance (m)


def next_rows(
    rows: np.ndarray, outcome: LaneOutcome, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    promote = outcome.success >= PROMOTE_SUCCESS
    demote = ~promote & (outcome.travel < DEMOTE_TRAVEL * outcome.cmd_distance)
    moved = rows + promote.astype(int) - demote.astype(int)
    solved = moved > TOP_ROW
    # legged_gym sends robots past the top row to a random one
    random_rows = rng.integers(0, TOP_ROW + 1, len(rows))
    new_rows = np.where(solved, random_rows, np.maximum(moved, 0))
    return new_rows, {
        "terrain_cl/promote_frac": float(promote.mean()),
        "terrain_cl/demote_frac": float(demote.mean()),
        "terrain_cl/top_reset_frac": float(solved.mean()),
    }


def row_stats(rows: np.ndarray) -> dict[str, float]:
    return {
        "terrain_cl/row_mean": float(rows.mean()),
        "terrain_cl/row_p10": float(np.percentile(rows, 10)),
        "terrain_cl/row_p90": float(np.percentile(rows, 90)),
    }


class TerrainLanes:
    """A uniform column genome per lane (own noise seed), drawn once, and
    the lane's current row."""

    def __init__(
        self, space: LevelSpace, rng: np.random.Generator, lanes: int
    ) -> None:
        self.space = space
        self.columns = space.uniform(rng, lanes)
        self.rows = rng.integers(0, MAX_INIT_ROW + 1, lanes)

    def levels(self) -> np.ndarray:
        return self.space.shrink(self.columns, self.rows / TOP_ROW)

    def advance(
        self, outcome: LaneOutcome, rng: np.random.Generator
    ) -> dict[str, float]:
        """Move every lane after its iteration; stats of the rows it ran."""
        stats = row_stats(self.rows)
        self.rows, moves = next_rows(self.rows, outcome, rng)
        return {**stats, **moves}
