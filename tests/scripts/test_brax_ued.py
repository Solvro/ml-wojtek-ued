from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.io import model as brax_model

from scripts.brax_ued import (
    Student,
    check_finite,
    last_finite_path,
    load_run,
    parse_args,
    run_spec,
    write_sidecar,
)
from utils.env import ObsSize

DEPLOY = ["--mode", "accel", "--name", "t", "--profile", "deploy"]
SIZE = ObsSize(46, 78)


def params() -> Any:
    env = SimpleNamespace(observation_size=SIZE, action_size=12)
    network = run_spec(parse_args(DEPLOY))["network"]
    student = Student(env, network, jax.random.key(0), 0.2)  # ty: ignore[invalid-argument-type]
    return student.inference_params()


def poisoned(tree: Any) -> Any:
    normalizer, policy, value = tree
    return normalizer, jax.tree.map(lambda x: x * jnp.nan, policy), value


def checkpoint(tmp_path: Path, tree: Any) -> Path:
    path = tmp_path / "run.pkl"
    write_sidecar(path, run_spec(parse_args(DEPLOY)), SIZE, 12)
    brax_model.save_params(str(path), tree)
    return path


def test_finite_params_pass_and_write_nothing(tmp_path: Path) -> None:
    good = params()
    path = checkpoint(tmp_path, good)
    check_finite(good, good, path, "iteration 0")
    assert not last_finite_path(path).exists()


def test_nan_params_fail_fast_and_keep_the_last_finite(tmp_path: Path) -> None:
    good = params()
    path = checkpoint(tmp_path, good)
    with pytest.raises(FloatingPointError, match="iteration 55"):
        check_finite(poisoned(good), good, path, "iteration 55")
    _, kept = load_run(last_finite_path(path))
    for ours, theirs in zip(jax.tree.leaves(kept), jax.tree.leaves(good), strict=True):
        np.testing.assert_array_equal(ours, theirs)


def test_nan_params_without_a_checkpoint_dir_still_fail() -> None:
    good = params()
    with pytest.raises(FloatingPointError, match="nothing saved"):
        check_finite(poisoned(good), good, None, "iteration 1")


def test_a_diverged_checkpoint_is_refused(tmp_path: Path) -> None:
    path = checkpoint(tmp_path, poisoned(params()))
    with pytest.raises(ValueError, match="non-finite"):
        load_run(path)


@pytest.mark.parametrize("flag", ["--command_box", "--arena"])
def test_eval_only_flags_cannot_reach_training(flag: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, flag, "benchmark"])
