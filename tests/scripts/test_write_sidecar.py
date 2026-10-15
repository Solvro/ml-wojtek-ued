import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest
from brax.io import model as brax_model
from brax.training.acme import running_statistics, specs

from scripts import write_sidecar
from scripts.brax_ued import parse_args, run_spec, sidecar_path
from utils.env import FunctionalEnv
from utils.env_overrides import merge_overrides

DEPLOY = ["--mode", "accel", "--name", "t", "--profile", "deploy"]
BENCHMARK = ["--mode", "accel", "--name", "t"]


def flat_env(spec: dict[str, Any], num_envs: int, train: bool) -> FunctionalEnv:
    # observation sizes do not depend on the arena, whose files tests lack
    overrides = merge_overrides(spec["env_overrides"], {"terrain": {"enable": False}})
    return FunctionalEnv(task=spec["task"], env_overrides=overrides)


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(write_sidecar, "make_env", flat_env)
    size = flat_env(run_spec(parse_args(DEPLOY)), 1, False).observation_size
    normalizer = running_statistics.init_state(
        {
            "state": specs.Array((size.actor,), jnp.dtype("float32")),
            "privileged_state": specs.Array((size.critic,), jnp.dtype("float32")),
        }
    )
    path = tmp_path / "old.pkl"
    brax_model.save_params(str(path), (normalizer,))
    return path


def test_backfill_writes_marked_sidecar(checkpoint: Path) -> None:
    path = write_sidecar.main(["--checkpoint", str(checkpoint), *DEPLOY])
    sidecar = json.loads(path.read_text())
    assert path == sidecar_path(checkpoint)
    assert sidecar["backfilled"] is True
    assert sidecar["profile"] == "deploy"


def test_backfill_with_other_run_flags_is_refused(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="different runs"):
        write_sidecar.main(["--checkpoint", str(checkpoint), *BENCHMARK])
    assert not sidecar_path(checkpoint).exists()


def test_backfill_keeps_existing_sidecar(checkpoint: Path) -> None:
    sidecar_path(checkpoint).write_text("{}")
    with pytest.raises(FileExistsError, match="delete it first"):
        write_sidecar.main(["--checkpoint", str(checkpoint), *DEPLOY])
    assert sidecar_path(checkpoint).read_text() == "{}"


def test_backfill_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        write_sidecar.main(["--checkpoint", str(tmp_path / "x.pkl"), *DEPLOY])
