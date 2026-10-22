from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from config.schema import Score
from training import curricula
from training.buffer import LevelBuffer
from training.progress import unvisited

Compose = Callable[..., Any]


def fake_run_levels(returns: list[float]) -> Callable[..., dict[str, Any]]:
    def run(_: Any, levels: np.ndarray, train: bool) -> dict[str, Any]:
        return {
            "score": np.full(len(levels), 7.0),
            "lane_episode_return": np.array(returns),
            "lane_roughness": np.zeros(len(levels)),
            "levels": levels,
        }

    return run


def accel_ctx(compose_cfg: Compose, mode: str) -> Any:
    cfg = compose_cfg("train", f"mode={mode}", "name=t", "num_lanes=2")
    space = SimpleNamespace(
        shape=(2,),
        mutate=lambda level, rng, grow_max: level + 1.0,
        easy=lambda rng, n, scale: np.zeros((n, 2)),
    )
    return SimpleNamespace(
        cfg=cfg,
        rng=np.random.default_rng(0),
        space=space,
        children=np.zeros((0, 2)),
        children_return=unvisited(0),
    )


@pytest.mark.parametrize("mode", ["accel_lp", "accel"])
def test_accel_lp_scores_against_the_previous_visit_and_the_parent(
    compose_cfg: Compose, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    ctx = accel_ctx(compose_cfg, mode)
    ctx.cfg.mode.replay_prob = 1.0
    buffer = LevelBuffer(8)
    levels = np.array([[0.0, 0.0], [5.0, 5.0]])
    buffer.add(levels, np.zeros(2), 0, np.zeros(2), np.array([1.0, 2.0]))
    monkeypatch.setattr(buffer, "sample", lambda *_: np.array([1, 0]))
    monkeypatch.setattr(curricula, "run_levels", fake_run_levels([3.0, 0.5]))
    replay = curricula.accel_iteration(ctx, buffer, 1)
    np.testing.assert_array_equal(buffer.last_return, [0.5, 3.0])
    np.testing.assert_array_equal(ctx.children_return, [3.0, 0.5])
    np.testing.assert_array_equal(ctx.children, [[6.0, 6.0], [1.0, 1.0]])

    ctx.cfg.mode.replay_prob = 0.0
    monkeypatch.setattr(curricula, "run_levels", fake_run_levels([2.0, 2.0]))
    explore = curricula.accel_iteration(ctx, buffer, 2)
    np.testing.assert_array_equal(buffer.last_return[-2:], [2.0, 2.0])
    if ctx.cfg.mode.score == Score.lp:
        np.testing.assert_allclose(replay["score"], [1.0, -0.5])
        np.testing.assert_allclose(explore["score"], [-1.0, 1.5])
        np.testing.assert_allclose(buffer.scores, [-0.5, 1.0, -1.0, 1.5])
    else:
        np.testing.assert_array_equal(explore["score"], [7.0, 7.0])


def test_fresh_explore_levels_score_zero_under_lp(
    compose_cfg: Compose, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = accel_ctx(compose_cfg, "accel_lp")
    monkeypatch.setattr(curricula, "run_levels", fake_run_levels([0.3, 0.9]))
    out = curricula.accel_iteration(ctx, LevelBuffer(8), 0)
    np.testing.assert_array_equal(out["score"], [0.0, 0.0])
