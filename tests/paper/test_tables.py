import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from config.schema import Robot
from paper.runs import run_prefixes
from paper.tables import tables


def test_run_prefixes_keeps_the_robots_set() -> None:
    prefixes = {Robot.wojtek: "ued_wojtek", Robot.go1: None}
    assert run_prefixes(prefixes) == {"wojtek": "ued_wojtek"}


def test_run_prefixes_refuses_none_set() -> None:
    with pytest.raises(ValueError, match=r"prefixes\.<robot>=<prefix>"):
        run_prefixes({Robot.wojtek: None, Robot.go1: None})


def test_tables_need_an_eval_dir(compose_cfg: Callable[..., Any]) -> None:
    cfg = compose_cfg("paper_tables")
    with pytest.raises(ValueError, match="eval_dir"):
        tables(cfg)


def write_eval(eval_dir: Path, run: str, success: float) -> None:
    metrics = {
        "random/d0.25/success_rate": success,
        "random/d0.25/cvar0.25_success": success / 2,
        "suite/success_rate": success,
        "flat/success_rate": 1.0,
        "param/uniform/success_rate": success / 10,
        "param/uniform/cvar0.25_success": 0.0,
    }
    (eval_dir / f"{run}_final_eval.json").write_text(json.dumps(metrics))


def test_main_table_reads_d025_and_uniform_goes_to_its_own_table(
    tmp_path: Path, compose_cfg: Callable[..., Any]
) -> None:
    for method, success in (("accel", 0.55), ("dr", 0.21)):
        write_eval(tmp_path, f"ued_go1_{method}_seed0", success)
    out = tmp_path / "tables"
    cfg = compose_cfg(
        "paper_tables",
        f"eval_dir={tmp_path}",
        f"out={out}",
        *[f"prefixes.{r}=null" for r in Robot if r != Robot.go1],
    )
    with pytest.warns(UserWarning):
        tables(cfg)
    main = (out / "main.tex").read_text()
    assert r"$d{=}0.25$" in main
    assert r"\textbf{55.0" in main
    assert "Uniform" not in main
    uniform = (out / "uniform.tex").read_text()
    assert "Uniform" in uniform
    assert r"\textbf{5.5" in uniform
