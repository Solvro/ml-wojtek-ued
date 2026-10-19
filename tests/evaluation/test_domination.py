import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evaluation.domination import (
    MethodRates,
    compare,
    compare_pair,
    domination_matrix,
    histogram,
    noise_abs_diff,
    solve_rate,
)

Compose = Callable[..., Any]
REPO = Path(__file__).resolve().parents[2]
PASSES, EPISODES = 2, 4
GENOMES = np.random.default_rng(0).random((6, 9))


def write_dump(
    eval_dir: Path,
    run: str,
    solved: Sequence[int],
    genomes: np.ndarray = GENOMES,
) -> None:
    """A tiny eval_brax.py npz: level i solved on the first solved[i] of
    its 8 episodes, the rest fall."""
    flat = np.arange(PASSES * EPISODES)[None] >= np.asarray(solved)[:, None]
    fell = flat.reshape(len(solved), PASSES, EPISODES).transpose(1, 0, 2)
    np.savez(
        eval_dir / f"{run}_final_eval.npz",
        param_uniform_fell=fell.astype(np.float32),
        param_uniform_travel=np.full(fell.shape, 3.0, np.float32),
        param_uniform_levels=genomes,
        random_levels=genomes,
    )


def rates(
    values: Sequence[float] | np.ndarray, episodes: int = 8
) -> MethodRates:
    return MethodRates([0], episodes, np.asarray(values, dtype=float))


def test_solve_rate_needs_no_fall_and_the_travel() -> None:
    fell = np.array([[[0, 1, 0, 0]], [[0, 0, 0, 0]]])
    travel = np.array([[[2.0, 5.0, 0.5, 2.0]], [[2.0, 2.0, 2.0, 0.9]]])
    assert solve_rate(fell, travel, 1.0) == pytest.approx([5 / 8])


def test_threshold_scales_with_the_robot(
    tmp_path: Path, compose_cfg: Compose
) -> None:
    # travel 3.0: past Wojtek's 1 m, short of Spot's 4.3 m
    for robot in ("wojtek", "spot"):
        for method in ("accel", "dr"):
            write_dump(tmp_path, f"{robot}_{method}_seed0", [8] * 6)
    cfg = compose_cfg(
        "domination",
        f"eval_dir={tmp_path}",
        "prefixes.wojtek=wojtek",
        "prefixes.spot=spot",
    )
    report = compare(cfg)
    assert report.robots["wojtek"].methods["accel"].mean_solve_rate == 1.0
    assert report.robots["spot"].methods["accel"].mean_solve_rate == 0.0


def test_rates_average_over_seeds(tmp_path: Path, compose_cfg: Compose) -> None:
    write_dump(tmp_path, "w_accel_seed0", [8, 0, 4, 2, 0, 0])
    write_dump(tmp_path, "w_accel_seed1", [0, 0, 4, 6, 0, 8])
    write_dump(tmp_path, "w_dr_seed0", [0] * 6)
    write_dump(tmp_path, "w_dr_seed1", [0] * 6)
    cfg = compose_cfg("domination", f"eval_dir={tmp_path}", "prefixes.wojtek=w")
    accel = compare(cfg).robots["wojtek"].methods["accel"]
    assert accel.seeds == [0, 1]
    assert accel.episodes_per_level == 16
    # per level (8+0)/16, 0, 8/16, 8/16, 0, 8/16
    assert accel.mean_solve_rate == pytest.approx(2 / 6)


def test_a_strictly_better_puts_all_mass_below_the_diagonal() -> None:
    a = np.array([0.95, 0.5, 0.35, 1.0])
    b = np.array([0.05, 0.25, 0.0, 0.5])
    counts = histogram(a, b, 10)
    assert counts.sum() == 4
    assert np.triu(counts).sum() == 0
    assert (counts[9, 0], counts[5, 2], counts[3, 0], counts[9, 5]) == (
        1,
        1,
        1,
        1,
    )


def test_identical_rates_sit_on_the_diagonal() -> None:
    # 12/40 * 10 floors to 2 in floats without the tolerance
    a = np.array([12 / 40, 0.0, 1.0, 0.1, 0.55])
    counts = histogram(a, a, 10)
    assert np.trace(counts) == 5
    assert counts[3, 3] == 1 and counts[9, 9] == 1 and counts[1, 1] == 1


def test_pair_fractions_and_ties() -> None:
    pair = compare_pair(
        ("accel", "dr"),
        rates([1.0, 0.5, 0.25, 0.0]),
        rates([0.5, 0.5, 0.75, 0.0]),
        10,
    )
    assert (pair.a_better, pair.b_better, pair.ties) == (0.25, 0.25, 0.5)
    assert pair.mean_diff == pytest.approx(0.0)
    assert pair.mean_diff_ci[0] <= 0.0 <= pair.mean_diff_ci[1]
    assert pair.mean_abs_diff == pytest.approx(0.25)
    assert np.sum(pair.counts) == 4


def test_mean_diff_ci_excludes_zero_when_a_wins_everywhere() -> None:
    a = rates(np.linspace(0.5, 1.0, 64))
    pair = compare_pair(("a", "b"), a, rates(np.linspace(0.0, 0.5, 64)), 10)
    assert pair.mean_diff == pytest.approx(0.5)
    assert pair.mean_diff_ci[0] > 0.4


def test_noise_reference_is_the_binomial_expectation() -> None:
    # one episode each at p = 0.5: |A - B| is 1 with probability 1/2
    half = np.array([0.5])
    assert noise_abs_diff(half, half, 1, 1) == pytest.approx(0.5)
    certain = np.array([0.0, 1.0])
    assert noise_abs_diff(certain, certain, 40, 40) == pytest.approx(0.0)
    assert noise_abs_diff(half, half, 40, 40) < noise_abs_diff(half, half, 8, 8)


def test_matrix_diagonal_is_zero_and_rows_complement() -> None:
    matrix = domination_matrix(
        {
            "accel": np.array([1.0, 0.5, 0.25, 0.0]),
            "dr": np.array([0.5, 0.5, 0.75, 0.0]),
            "plr": np.array([0.0, 1.0, 0.25, 0.5]),
        }
    )
    better, ties = np.asarray(matrix.row_better), np.asarray(matrix.ties)
    assert np.all(np.diag(better) == 0)
    assert np.allclose(better + better.T + ties, 1.0)
    assert better[0, 1] == 0.25 and ties[0, 1] == 0.5


def test_mismatched_levels_refuse(tmp_path: Path, compose_cfg: Compose) -> None:
    write_dump(tmp_path, "w_accel_seed0", [8] * 6)
    write_dump(tmp_path, "w_dr_seed0", [8] * 6, GENOMES + 0.01)
    cfg = compose_cfg("domination", f"eval_dir={tmp_path}", "prefixes.wojtek=w")
    with pytest.raises(ValueError, match="different levels"):
        compare(cfg)


def test_missing_level_set_names_the_sets_present(
    tmp_path: Path, compose_cfg: Compose
) -> None:
    write_dump(tmp_path, "w_accel_seed0", [8] * 6)
    cfg = compose_cfg(
        "domination",
        f"eval_dir={tmp_path}",
        "prefixes.wojtek=w",
        "level_set=random_d1.0",
    )
    with pytest.raises(KeyError, match=r"random_d1\.0_fell.*param_uniform"):
        compare(cfg)


def test_missing_seeds_and_methods_are_reported(
    tmp_path: Path, compose_cfg: Compose
) -> None:
    for seed in (0, 1, 2):
        write_dump(tmp_path, f"w_accel_seed{seed}", [8] * 6)
    write_dump(tmp_path, "w_dr_seed0", [0] * 6)
    write_dump(tmp_path, "g_accel_seed0", [8] * 6)
    write_dump(tmp_path, "g_plr_seed0", [8] * 6)
    cfg = compose_cfg(
        "domination",
        f"eval_dir={tmp_path}",
        "prefixes.wojtek=w",
        "prefixes.go1=g",
    )
    with pytest.warns(UserWarning) as caught:
        report = compare(cfg)
    wojtek, go1 = report.robots["wojtek"], report.robots["go1"]
    assert wojtek.missing_seeds == {"dr": [1, 2]}
    assert wojtek.missing_methods == ["plr"]
    assert go1.missing_methods == ["dr"]
    assert go1.pair is None and wojtek.pair is not None
    messages = " ".join(str(w.message) for w in caught)
    assert "dr lacks seeds [1, 2]" in messages
    assert "no accel vs dr heatmap" in messages


def test_no_robot_with_the_pair_refuses(
    tmp_path: Path, compose_cfg: Compose
) -> None:
    write_dump(tmp_path, "w_accel_seed0", [8] * 6)
    cfg = compose_cfg("domination", f"eval_dir={tmp_path}", "prefixes.wojtek=w")
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="both"):
        compare(cfg)


def test_no_dumps_refuses(tmp_path: Path, compose_cfg: Compose) -> None:
    cfg = compose_cfg("domination", f"eval_dir={tmp_path}", "prefixes.wojtek=w")
    with pytest.raises(FileNotFoundError, match="w_<method>_seed<k>"):
        compare(cfg)


def test_cli_writes_json_and_figures(tmp_path: Path) -> None:
    evals, out = tmp_path / "evals", tmp_path / "out"
    evals.mkdir()
    for robot in ("w", "g"):
        for seed in (0, 1):
            write_dump(evals, f"{robot}_accel_seed{seed}", [8, 6, 4, 8, 2, 0])
            write_dump(evals, f"{robot}_dr_seed{seed}", [4, 6, 0, 2, 2, 0])
            write_dump(evals, f"{robot}_plr_seed{seed}", [0, 8, 4, 4, 2, 1])
    subprocess.run(
        [
            sys.executable,
            "src/scripts/compare_domination.py",
            f"eval_dir={evals}",
            f"out={out}",
            "prefixes.wojtek=w",
            "prefixes.go1=g",
            f"hydra.run.dir={tmp_path / 'hydra'}",
        ],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )
    report = json.loads((out / "domination.json").read_text())
    pair = report["robots"]["wojtek"]["pair"]
    assert (pair["a_better"], pair["b_better"]) == (0.5, 0.0)
    assert report["robots"]["go1"]["matrix"]["methods"] == [
        "dr",
        "plr",
        "accel",
    ]
    written = {p.name for p in out.iterdir()}
    assert {
        "domination_accel_vs_dr.pdf",
        "domination_accel_vs_dr_wojtek.png",
        "domination_accel_vs_dr_go1.pdf",
        "domination_matrix.pdf",
    } <= written
