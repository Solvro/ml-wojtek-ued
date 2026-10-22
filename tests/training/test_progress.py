import numpy as np

from training.progress import learning_progress, unvisited


def test_first_visit_scores_zero() -> None:
    lp = learning_progress(unvisited(2), np.array([4.0, -1.0]))
    np.testing.assert_array_equal(lp, [0.0, 0.0])


def test_score_is_the_signed_change_in_return() -> None:
    lp = learning_progress(np.array([1.0, 3.0]), np.array([2.5, 1.0]))
    np.testing.assert_allclose(lp, [1.5, -2.0])


def test_a_child_compares_against_its_parent_and_fresh_levels_score_zero() -> (
    None
):
    prior = np.concatenate([np.array([2.0]), unvisited(1)])
    lp = learning_progress(prior, np.array([1.5, 9.0]))
    np.testing.assert_allclose(lp, [-0.5, 0.0])
