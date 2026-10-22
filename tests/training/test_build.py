from collections.abc import Callable
from typing import Any

import pytest

from training.build import check_config

Compose = Callable[..., Any]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (("mode=accel", "mode.bc=uni"), "two students"),
        (("mode=dr", "mode.bc=bi"), "two students"),
        (("mode=paired_uni", "mode.bc_every=0"), "bc_every"),
        (("mode=paired_evo", "mode.score=learnability"), "PAIRED regret"),
        (("mode=paired", "level_steps=30"), "multiple"),
    ],
)
def test_check_config_rejects(
    compose_cfg: Compose, overrides: tuple[str, ...], match: str
) -> None:
    cfg = compose_cfg("train", "name=t", *overrides)
    with pytest.raises(ValueError, match=match):
        check_config(cfg)
