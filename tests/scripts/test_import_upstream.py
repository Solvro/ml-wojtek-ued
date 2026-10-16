import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from ml_collections import config_dict

from scripts import export_policy, import_upstream
from scripts.brax_ued import (
    Student,
    load_run,
    make_network,
    parse_args,
    run_spec,
)
from utils.env import ObsSize

DEPLOY = ["--mode", "accel", "--name", "imported", "--profile", "deploy"]
STEP = 123456


def deploy_env() -> Any:
    return export_policy.build_env(run_spec(parse_args(DEPLOY)))


def deploy_size() -> ObsSize:
    obs = deploy_env().reset(jax.random.key(0)).obs
    return ObsSize(obs["state"].shape[-1], obs["privileged_state"].shape[-1])


def fake_upstream(
    tmp_path: Path,
    critic: int | None = None,
    hidden: list[int] | None = None,
    env_edit: dict[str, Any] | None = None,
) -> Path:
    """A w01-tek run dir: run.json + checkpoints/<step> as train.py saves it."""
    spec = run_spec(parse_args(DEPLOY))
    size = deploy_size()
    size = ObsSize(size.actor, critic or size.critic)
    network = dict(spec["network"])
    if hidden:
        network["policy_hidden_layer_sizes"] = hidden
    nets = make_network(network, size, 12)
    keys = jax.random.split(jax.random.key(7), 3)
    normalizer = running_statistics.init_state(
        {
            "state": specs.Array((size.actor,), jnp.dtype("float32")),
            "privileged_state": specs.Array((size.critic,), jnp.dtype("float32")),
        }
    )
    params = [
        normalizer,
        nets.policy_network.init(keys[0]),
        nets.value_network.init(keys[1]),
    ]
    config = config_dict.ConfigDict(
        {
            "observation_size": {
                "state": {"shape": [size.actor]},
                "privileged_state": {"shape": [size.critic]},
            },
            "action_size": 12,
            "normalize_observations": True,
            "network_factory_kwargs": network,
        }
    )
    run_dir = tmp_path / "runs" / "upstream"
    ppo_checkpoint.save(str(run_dir / "checkpoints"), STEP, params, config)
    env_config = json.loads(json.dumps(deploy_env()._config.to_dict()))
    env_config.update(env_edit or {})
    (run_dir / "run.json").write_text(
        json.dumps({"run_name": "upstream", "env_config": env_config})
    )
    return run_dir / "checkpoints" / f"{STEP:012d}"


def run_import(checkpoint: Path, out: Path, flags: list[str] = DEPLOY) -> Any:
    return import_upstream.main(
        ["--checkpoint", str(checkpoint), "--out", str(out), *flags]
    )


def test_import_round_trips_through_export(tmp_path: Path) -> None:
    out = tmp_path / "imported.pkl"
    spec = run_import(fake_upstream(tmp_path), out)
    assert spec["profile"] == "deploy"
    assert spec["imported_from"]["env_steps"] == STEP
    _, params = load_run(out)
    upstream = ppo_checkpoint.load(str(fake_upstream(tmp_path / "again")))
    for ours, theirs in zip(
        jax.tree.leaves(params[1:]), jax.tree.leaves(upstream[1:]), strict=True
    ):
        np.testing.assert_array_equal(ours, theirs)
    meta = export_policy.export(out, tmp_path / "deploy")
    assert meta["schema_version"] == 2
    assert (tmp_path / "deploy" / "policy.npz").exists()


def test_consumed_key_mismatch_is_refused(tmp_path: Path) -> None:
    checkpoint = fake_upstream(tmp_path, env_edit={"action_scale": [0.5, 0.5, 0.5]})
    out = tmp_path / "imported.pkl"
    with pytest.raises(ValueError, match="action_scale"):
        run_import(checkpoint, out)
    assert not out.exists()


def test_critic_size_mismatch_is_refused(tmp_path: Path) -> None:
    checkpoint = fake_upstream(tmp_path, critic=deploy_size().critic + 8)
    with pytest.raises(ValueError, match="critic obs"):
        run_import(checkpoint, tmp_path / "imported.pkl")


def test_network_mismatch_is_refused(tmp_path: Path) -> None:
    checkpoint = fake_upstream(tmp_path, hidden=[256, 128])
    with pytest.raises(ValueError, match="policy_hidden_layer_sizes"):
        run_import(checkpoint, tmp_path / "imported.pkl")


def test_benchmark_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--profile deploy"):
        run_import(fake_upstream(tmp_path), tmp_path / "x.pkl", DEPLOY[:4])


def test_checkpoint_without_run_json_is_refused(tmp_path: Path) -> None:
    checkpoint = fake_upstream(tmp_path)
    (checkpoint.parents[1] / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="run.json"):
        run_import(checkpoint, tmp_path / "x.pkl")


def student(size: ObsSize) -> Student:
    env = SimpleNamespace(observation_size=size, action_size=12)
    spec = run_spec(parse_args(DEPLOY))
    return Student(env, spec["network"], jax.random.key(1), 0.3)  # ty: ignore[invalid-argument-type]


def test_warm_start_takes_every_imported_param(tmp_path: Path) -> None:
    out = tmp_path / "imported.pkl"
    run_import(fake_upstream(tmp_path), out)
    _, params = load_run(out)
    s = student(deploy_size())
    s.warm_start(params, str(out))
    for ours, theirs in zip(
        jax.tree.leaves(s.inference_params()), jax.tree.leaves(params), strict=True
    ):
        np.testing.assert_array_equal(ours, theirs)


def test_warm_start_refuses_other_shapes(tmp_path: Path) -> None:
    out = tmp_path / "imported.pkl"
    run_import(fake_upstream(tmp_path), out)
    _, params = load_run(out)
    size = deploy_size()
    with pytest.raises(ValueError, match="value shapes"):
        student(ObsSize(size.actor, size.critic + 4)).warm_start(params, str(out))


def test_default_run_has_no_init_from() -> None:
    args = parse_args(DEPLOY)
    assert args.init_from is None
    assert run_spec(args)["init_from"] is None


def test_init_from_is_recorded_in_the_sidecar_spec() -> None:
    spec = run_spec(parse_args([*DEPLOY, "--init_from", "a1long.pkl"]))
    assert spec["init_from"] == "a1long.pkl"
