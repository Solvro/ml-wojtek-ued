from collections.abc import Callable
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
