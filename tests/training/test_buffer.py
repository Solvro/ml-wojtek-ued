import numpy as np

from training.buffer import LevelBuffer


def filled(scores: list[float], capacity: int = 8) -> LevelBuffer:
    n = len(scores)
    buffer = LevelBuffer(capacity)
    levels = np.arange(n, dtype=float)[:, None]
    returns = 10.0 * np.arange(n)
    buffer.add(levels, np.array(scores), 0, np.zeros(n), returns)
    return buffer


def test_eviction_keeps_each_level_with_its_last_return() -> None:
    buffer = filled([0.5, -2.0, 0.0, 1.0], capacity=3)
    kept = sorted(zip(buffer.scores, np.concatenate(buffer.levels)))
    assert kept == [(0.0, 2.0), (0.5, 0.0), (1.0, 3.0)]
    np.testing.assert_array_equal(
        buffer.last_return, 10.0 * np.concatenate(buffer.levels)
    )


def test_update_overwrites_score_and_last_return() -> None:
    buffer = filled([0.0, 0.0])
    buffer.update(np.array([1]), np.array([-0.3]), 4, np.array([7.0]))
    np.testing.assert_array_equal(buffer.scores, [0.0, -0.3])
    np.testing.assert_array_equal(buffer.last_return, [0.0, 7.0])
    np.testing.assert_array_equal(buffer.last_seen, [0, 4])


def test_negative_scores_are_replayed_least() -> None:
    buffer = filled([-1.0, 0.0, 2.0])
    buffer.staleness = 0.0
    draws = buffer.sample(np.random.default_rng(0), 3000, 1)
    counts = np.bincount(draws, minlength=3)
    assert counts[0] < counts[1] < counts[2]


def test_a_full_buffer_of_fresh_zero_scores_still_samples_every_level() -> None:
    buffer = filled([0.0] * 6, capacity=4)
    assert len(buffer) == 4
    draws = buffer.sample(np.random.default_rng(0), 4000, 1)
    assert set(draws) == {0, 1, 2, 3}
