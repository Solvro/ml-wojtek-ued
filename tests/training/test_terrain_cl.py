import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from hfield.level_space import make_level_space
from training.terrain_cl import (
    MAX_INIT_ROW,
    TOP_ROW,
    LaneOutcome,
    TerrainLanes,
    next_rows,
)


def outcome(
    success: list[float], travel: list[float], cmd: list[float]
) -> LaneOutcome:
    return LaneOutcome(np.array(success), np.array(travel), np.array(cmd))


def test_half_the_envs_succeeding_promotes() -> None:
    rows, stats = next_rows(
        np.array([3, 3]),
        outcome([0.5, 0.49], [9.0, 9.0], [1.0, 1.0]),
        np.random.default_rng(0),
    )
    assert rows.tolist() == [4, 3]
    assert stats["terrain_cl/promote_frac"] == 0.5
    assert stats["terrain_cl/demote_frac"] == 0.0


def test_short_travel_demotes_unless_promoted() -> None:
    # commanded 2 m: below 1 m of travel demotes, success outranks it
    rows, stats = next_rows(
        np.array([3, 3, 3]),
        outcome([0.0, 0.0, 0.6], [0.99, 1.0, 0.1], [2.0, 2.0, 2.0]),
        np.random.default_rng(0),
    )
    assert rows.tolist() == [2, 3, 4]
    assert stats["terrain_cl/demote_frac"] == pytest.approx(1 / 3)


def test_demotion_clips_at_row_zero() -> None:
    rows, _ = next_rows(
        np.array([0]), outcome([0.0], [0.0], [1.0]), np.random.default_rng(0)
    )
    assert rows.tolist() == [0]


def test_promotion_past_the_top_row_resets_to_a_random_row() -> None:
    n = 2000
    rows, stats = next_rows(
        np.full(n, TOP_ROW),
        LaneOutcome(np.ones(n), np.zeros(n), np.ones(n)),
        np.random.default_rng(0),
    )
    assert stats["terrain_cl/top_reset_frac"] == 1.0
    assert set(rows.tolist()) == set(range(TOP_ROW + 1))


def test_top_row_stays_without_promotion() -> None:
    rows, stats = next_rows(
        np.array([TOP_ROW]),
        outcome([0.2], [5.0], [1.0]),
        np.random.default_rng(0),
    )
    assert rows.tolist() == [TOP_ROW]
    assert stats["terrain_cl/top_reset_frac"] == 0.0


@pytest.mark.parametrize("kind", ["param", "grid"])
def test_lane_level_is_its_column_shrunk_by_row_over_ten(kind: str) -> None:
    space = make_level_space(kind, HFieldConfig(coarse_n=5), 2, 0.1)
    lanes = TerrainLanes(space, np.random.default_rng(0), 3)
    columns = lanes.columns.copy()
    lanes.rows = np.array([0, 4, 10])
    levels = lanes.levels()
    flat = space.shrink(columns, np.zeros(3))
    np.testing.assert_allclose(levels[0], flat[0])
    np.testing.assert_allclose(
        levels[1], flat[1] + 0.4 * (columns[1] - flat[1])
    )
    np.testing.assert_allclose(levels[2], columns[2])
    np.testing.assert_array_equal(lanes.columns, columns)


def test_param_lanes_keep_their_noise_seed() -> None:
    space = make_level_space("param", HFieldConfig(), 2, 0.1)
    lanes = TerrainLanes(space, np.random.default_rng(0), 4)
    seeds = lanes.columns[:, -1]
    assert len(set(seeds.tolist())) == 4
    np.testing.assert_array_equal(lanes.levels()[:, -1], seeds)


def test_initial_rows_are_uniform_up_to_max_init_row() -> None:
    space = make_level_space("param", HFieldConfig(), 2, 0.1)
    lanes = TerrainLanes(space, np.random.default_rng(0), 600)
    assert set(lanes.rows.tolist()) == set(range(MAX_INIT_ROW + 1))


def test_advance_logs_the_rows_it_ran_then_moves() -> None:
    space = make_level_space("param", HFieldConfig(), 2, 0.1)
    lanes = TerrainLanes(space, np.random.default_rng(0), 2)
    lanes.rows = np.array([2, 4])
    stats = lanes.advance(
        outcome([1.0, 0.0], [0.0, 0.0], [1.0, 1.0]), np.random.default_rng(0)
    )
    assert stats["terrain_cl/row_mean"] == 3.0
    assert lanes.rows.tolist() == [3, 3]
