import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from config.schema import Robot
from paper.stats import stats

D025 = "random/d0.25/success_rate"
SUITE = "suite/success_rate"


def write_eval(
    eval_dir: Path, robot: str, method: str, seed: int, success: float
) -> None:
    run = eval_dir / f"ued_{robot}_{method}_seed{seed}_final_eval"
    metrics = {D025: success, SUITE: success / 2}
    run.with_suffix(".json").write_text(json.dumps(metrics))
    np.savez(run.with_suffix(".npz"))


def write_method(
    eval_dir: Path, robot: str, method: str, values: list[float]
) -> None:
    for seed, value in enumerate(values):
        write_eval(eval_dir, robot, method, seed, value)


def run_stats(
    compose_cfg: Callable[..., Any], eval_dir: Path, robots: list[str]
) -> dict[str, Any]:
    out = eval_dir / "stats"
    cfg = compose_cfg(
        "paper_stats",
        f"eval_dir={eval_dir}",
        f"out={out}",
        "reps=200",
        "poi_reps=50",
        *[f"prefixes.{r}=null" for r in Robot if r not in robots],
    )
    stats(cfg)
    return json.loads((out / "stats.json").read_text())


@pytest.fixture
def report(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> Iterator[dict[str, Any]]:
    """ACCEL dominates DR on both robots; PLR copies DR."""
    spread = [0.0, 0.2, 0.3, 0.4, 0.5]
    for robot in ("go1", "spot"):
        write_method(tmp_path, robot, "accel", [0.6, 0.7, 0.8, 0.9, 1.0])
        write_method(tmp_path, robot, "dr", spread)
        write_method(tmp_path, robot, "plr", spread)
    yield run_stats(compose_cfg, tmp_path, ["go1", "spot"])


def test_iqm_is_the_mean_of_the_middle_seeds(report: dict[str, Any]) -> None:
    dr = report["metrics"][D025]["per_robot"]["go1"]["dr"]
    assert dr["iqm"] == pytest.approx(0.3)
    assert dr["mean"] == pytest.approx(0.28)
    assert dr["seeds"] == [0, 1, 2, 3, 4]


def test_every_ci_contains_its_point_estimate(report: dict[str, Any]) -> None:
    for values in report["metrics"].values():
        estimates = [
            *values["aggregate"].values(),
            *(e for by in values["per_robot"].values() for e in by.values()),
        ]
        for e in estimates:
            for stat in ("iqm", "mean"):
                lo, hi = e[f"{stat}_ci"]
                assert lo <= e[stat] <= hi
        for by in values["poi"].values():
            for e in by.values():
                assert e["ci"][0] <= e["p"] <= e["ci"][1]


def test_aggregate_is_min_max_normalised_per_robot(
    report: dict[str, Any],
) -> None:
    values = report["metrics"][D025]
    assert values["normalisation"]["go1"]["max_method"] == "accel"
    aggregate = values["aggregate"]
    assert aggregate["accel"]["mean"] == pytest.approx(1.0)
    assert aggregate["dr"]["mean"] == pytest.approx(0.0)
    assert aggregate["dr"]["runs"] == 10


def test_probability_of_improvement(report: dict[str, Any]) -> None:
    poi = report["metrics"][D025]["poi"]
    assert poi["accel"]["dr"]["p"] == 1.0
    assert poi["dr"]["accel"]["p"] == 0.0
    assert poi["plr"]["dr"]["p"] == 0.5
    assert poi["plr"]["dr"]["robots"] == ["go1", "spot"]


def test_named_pairs_skip_methods_without_runs(
    report: dict[str, Any],
) -> None:
    values = report["metrics"][D025]
    named = [(p["x"], p["y"]) for p in values["named_pairs"]]
    assert named == [("accel", "dr"), ("plr", "dr")]
    assert ["accel_sfl", "accel"] in values["skipped_pairs"]


def test_outputs_are_written(report: dict[str, Any], tmp_path: Path) -> None:
    out = tmp_path / "stats"
    assert (out / "iqm.pdf").exists()
    assert (out / "poi.pdf").exists()
    table = (out / "stats.tex").read_text()
    rows = [line.split(" & ")[0] for line in table.splitlines()[4:7]]
    assert rows == ["DR", r"PLR$^{\perp}$", "ACCEL"]
    assert r"Method & IQM $d{=}0.25$ & IQM OOD suite \\" in table


def test_refuses_a_robot_where_every_method_ties(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> None:
    write_method(tmp_path, "go1", "accel", [0.2, 0.4])
    write_method(tmp_path, "go1", "dr", [0.4, 0.2])
    with pytest.raises(ValueError, match="prefixes.go1=null"):
        run_stats(compose_cfg, tmp_path, ["go1"])


def test_refuses_a_robot_without_the_metric(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> None:
    run = tmp_path / "ued_go1_dr_seed0_final_eval"
    run.with_suffix(".json").write_text(json.dumps({SUITE: 0.5}))
    np.savez(run.with_suffix(".npz"))
    with (
        pytest.warns(UserWarning, match=f"no {D025}"),
        pytest.raises(ValueError, match="no run has"),
    ):
        run_stats(compose_cfg, tmp_path, ["go1"])


def test_a_method_missing_on_a_robot_is_left_out_of_the_aggregate(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> None:
    for robot in ("go1", "spot"):
        write_method(tmp_path, robot, "dr", [0.1, 0.2])
        write_method(tmp_path, robot, "accel", [0.5, 0.6])
    write_method(tmp_path, "go1", "plr", [0.3, 0.4])
    with pytest.warns(UserWarning, match=r"plr has no runs on \['spot'\]"):
        report = run_stats(compose_cfg, tmp_path, ["go1", "spot"])
    values = report["metrics"][D025]
    assert "plr" in values["per_robot"]["go1"]
    assert "plr" not in values["aggregate"]
    assert values["missing"] == {"plr": ["spot"]}
    assert values["poi"]["plr"]["dr"]["robots"] == ["go1"]
    table = (tmp_path / "stats" / "stats.tex").read_text()
    assert r"PLR$^{\perp}$ & -- & -- \\" in table


def test_unequal_seed_counts_keep_the_common_count(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> None:
    write_method(tmp_path, "go1", "dr", [0.1, 0.2, 0.3])
    write_method(tmp_path, "go1", "accel", [0.5, 0.6, 0.7])
    write_method(tmp_path, "spot", "dr", [0.1, 0.2])
    write_method(tmp_path, "spot", "accel", [0.5, 0.6, 0.7])
    with pytest.warns(UserWarning, match="dr: 1 runs dropped"):
        report = run_stats(compose_cfg, tmp_path, ["go1", "spot"])
    values = report["metrics"][D025]
    assert values["aggregate"]["dr"]["runs"] == 4
    assert values["per_robot"]["go1"]["dr"]["runs"] == 3
