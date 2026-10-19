from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from evaluation import sim2sim as s2s
from evaluation.suite import SUITE_STRENGTHS, SUITE_TERRAINS, suite_hfield
from hfield.level_space import make_level_space
from robots.scale import ROBOT_SCALE, hfield_config


def level_cfg(compose_cfg: Callable[..., Any], robot: str = "wojtek") -> Any:
    return compose_cfg(
        "sim2sim",
        "checkpoint=c",
        "name=n",
        "out_dir=o",
        "num_levels=8",
        "suite_instances=3",
        f"robot={robot}",
    )


def test_level_sets_names_and_sizes(compose_cfg: Callable[..., Any]) -> None:
    cfg = level_cfg(compose_cfg)
    config = hfield_config(cfg.robot.name, cfg.arena)
    sets = s2s.level_sets(cfg, config)
    cells = [f"suite/{t}_{k}" for t in SUITE_TERRAINS for k in (1, 2)]
    assert list(sets) == ["flat", "param/uniform", *cells]
    assert sets["param/uniform"].shape == (8, config.grid_size**2)
    assert all(sets[c].shape == (3, config.grid_size**2) for c in cells)
    np.testing.assert_allclose(sets["flat"], 0.5, atol=1e-6)


def test_uniform_levels_are_eval_brax_prefix(
    compose_cfg: Callable[..., Any],
) -> None:
    cfg = level_cfg(compose_cfg)
    config = hfield_config(cfg.robot.name, cfg.arena)
    space = make_level_space("param", config, n_edits=1, edit_scale=0.0)
    eval_brax = space.uniform(np.random.default_rng(cfg.seed + 1), 2048)
    np.testing.assert_array_equal(
        s2s.level_sets(cfg, config)["param/uniform"],
        space.render(eval_brax[:8]),
    )


@pytest.mark.parametrize("robot", ["wojtek", "go1"])
def test_suite_cells_match_eval_brax(
    compose_cfg: Callable[..., Any], robot: str
) -> None:
    cfg = level_cfg(compose_cfg, robot)
    config = hfield_config(cfg.robot.name, cfg.arena)
    rng = np.random.default_rng(cfg.seed)
    scale = ROBOT_SCALE[robot]
    eval_brax = {
        f"suite/{t}_{k}": np.stack(
            [suite_hfield(config, t, k, rng, scale) for _ in range(3)]
        )
        for t in SUITE_TERRAINS
        for k in SUITE_STRENGTHS
    }
    sets = s2s.level_sets(cfg, config)
    for name, hfields in sets.items():
        if name.startswith("suite/"):
            np.testing.assert_array_equal(hfields, eval_brax[name])


def test_summarize_reports_drop_against_mjx_match() -> None:
    fell = np.zeros((4, 2))
    travel = np.array([[2.0, 2.0], [2.0, 2.0], [0.5, 2.0], [0.5, 0.5]])
    worse = travel.copy()
    worse[0] = 0.0
    sets = {"flat": (fell, travel, 0), "suite/waves_1": (fell, travel, 0)}
    results = {
        "mjx_match": sets,
        "soft_contact": {
            "flat": (fell, worse, 2),
            "suite/waves_1": sets["flat"],
        },
    }
    metrics = s2s.summarize(results, threshold=1.0)
    assert metrics["mjx_match/flat/success_rate"] == pytest.approx(0.625)
    assert "mjx_match/flat/drop" not in metrics
    assert metrics["soft_contact/flat/drop"] == pytest.approx(0.25)
    assert metrics["soft_contact/suite/drop"] == 0.0
    assert metrics["soft_contact/flat/diverged_per_episode"] == pytest.approx(
        0.25
    )
    assert metrics["mjx_match/suite/waves_1/success_rate"] == pytest.approx(
        0.625
    )
