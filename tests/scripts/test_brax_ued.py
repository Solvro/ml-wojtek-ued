import math
from collections.abc import Callable, Iterator
from typing import Any

import jax
import pytest

from config.schema import BC, ModeConfig
from scripts import brax_ued

Compose = Callable[..., Any]

# the CPU smoke run: 2 levels x 16 envs, one 20-step unroll per iteration,
# 3 iterations
TINY = (
    "num_lanes=2",
    "envs_per_lane=16",
    "level_steps=20",
    "eval_levels=2",
    "eval_every=1",
    "num_timesteps=1500",
    "mode.buffer_size=8",
    "backend=jax",
)


def mode(bc: BC, bc_every: int = 1) -> ModeConfig:
    return ModeConfig(bc=bc, bc_coef=0.5, bc_every=bc_every)


@pytest.mark.parametrize(
    ("bc", "it", "expected"),
    [
        (BC.none, 0, (None, None)),
        (BC.uni, 0, (0.5, None)),
        (BC.bi, 0, (0.5, 0.5)),
        (BC.bi, 3, (0.5, 0.5)),
    ],
)
def test_bc_coefs_per_variant(bc: BC, it: int, expected: tuple) -> None:
    assert brax_ued.bc_coefs(mode(bc), it) == expected


def test_bc_coefs_zero_between_distillation_iterations() -> None:
    assert brax_ued.bc_coefs(mode(BC.bi, bc_every=3), 4) == (0.0, 0.0)
    assert brax_ued.bc_coefs(mode(BC.uni, bc_every=3), 6) == (0.5, None)


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
        brax_ued.check_config(cfg)


@pytest.fixture(scope="module")
def compile_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    # the variants share most of their programs (env, eval, the unroll)
    jax.config.update(
        "jax_compilation_cache_dir", str(tmp_path_factory.mktemp("jax_cache"))
    )
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    yield
    jax.config.update("jax_compilation_cache_dir", None)


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def log(self, row: dict[str, Any]) -> None:
        self.logs.append(row)

    def finish(self) -> None:
        pass


@pytest.mark.slow
@pytest.mark.parametrize(
    ("mode_name", "bc"),
    [
        ("paired", "none"),
        ("paired", "uni"),
        ("paired", "bi"),
        ("paired_evo", "none"),
    ],
)
@pytest.mark.usefixtures("compile_cache")
def test_paired_variant_cpu_smoke(
    compose_cfg: Compose,
    monkeypatch: pytest.MonkeyPatch,
    mode_name: str,
    bc: str,
) -> None:
    run = FakeRun()
    monkeypatch.setattr(brax_ued.wandb, "init", lambda **_: run)
    cfg = compose_cfg("train", f"mode={mode_name}", f"mode.bc={bc}", "name=t", *TINY)
    brax_ued.train(cfg)
    assert len(run.logs) == 3
    for row in run.logs:
        assert math.isfinite(row["prot_bc_kl"]) and math.isfinite(row["antag_bc_kl"])
        assert row["sim_steps"] == 2 * row["env_steps"]
        assert row["bc_coef"] == (0.0 if bc == "none" or not row["replay"] else 0.5)
    if mode_name == "paired_evo":
        assert {row["replay"] for row in run.logs} == {0, 1}
        assert run.logs[-1]["buffer_size"] > 0
