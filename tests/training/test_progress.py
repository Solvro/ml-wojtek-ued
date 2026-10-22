import numpy as np
import pytest

from config.schema import ModeConfig
from training.progress import learning_progress, unvisited, update_progress

MODE = ModeConfig(lp_fast_rate=0.5, lp_slow_rate=0.1)


def test_first_visit_starts_both_emas_at_success_with_score_zero() -> None:
    progress = update_progress(unvisited(2), np.array([0.25, 1.0]), MODE)
    np.testing.assert_array_equal(progress, [[0.25, 0.25], [1.0, 1.0]])
    np.testing.assert_array_equal(learning_progress(progress), [0.0, 0.0])


def test_later_visits_move_each_ema_by_its_rate() -> None:
    progress = update_progress(np.array([[0.2, 0.2]]), np.array([1.0]), MODE)
    np.testing.assert_allclose(progress, [[0.6, 0.28]])
    assert learning_progress(progress)[0] == pytest.approx(0.32)


def test_score_is_absolute_so_forgetting_counts_too() -> None:
    progress = update_progress(np.array([[0.8, 0.8]]), np.array([0.0]), MODE)
    np.testing.assert_allclose(progress, [[0.4, 0.72]])
    assert learning_progress(progress)[0] == pytest.approx(0.32)


def test_children_inherit_parent_emas_and_fresh_levels_start_over() -> None:
    parent = np.array([[0.6, 0.28]])
    prior = np.concatenate([parent, unvisited(1)])
    progress = update_progress(prior, np.array([0.6, 0.6]), MODE)
    np.testing.assert_allclose(progress, [[0.6, 0.312], [0.6, 0.6]])
