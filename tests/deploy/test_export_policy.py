import dataclasses
import json
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.io import model as brax_model
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from deploy.runtime import WojtekPolicy
from scripts import export_policy
from scripts.brax_ued import (
    load_run,
    make_network,
    parse_args,
    run_spec,
    sidecar_path,
    write_sidecar,
)
from utils.env import ObsSize

SNAPSHOT = json.loads(
    (Path(__file__).parent / "upstream_terrain_blind_v4_4d.json").read_text()
)


def profile_spec(profile: str) -> dict[str, Any]:
    flags = ["--mode", "accel", "--name", "t", "--profile", profile]
    return run_spec(parse_args(flags))


def obs_size(env: Any) -> ObsSize:
    obs = env.reset(jax.random.key(0)).obs
    return ObsSize(obs["state"].shape[-1], obs["privileged_state"].shape[-1])


def random_params(network: Any, size: ObsSize, seed: int) -> tuple:
    keys = jax.random.split(jax.random.key(seed), 4)
    normalizer = running_statistics.init_state(
        {
            "state": specs.Array((size.actor,), jnp.dtype("float32")),
            "privileged_state": specs.Array((size.critic,), jnp.dtype("float32")),
        }
    )
    # a trained-looking normalizer: the export must carry mean and std over
    mean = cast(dict, normalizer.mean)
    std = cast(dict, normalizer.std)
    normalizer = dataclasses.replace(
        normalizer,
        mean={**mean, "state": jax.random.normal(keys[0], (size.actor,))},
        std={
            **std,
            "state": jax.random.uniform(keys[1], (size.actor,), minval=0.2, maxval=3),
        },
    )
    policy = network.policy_network.init(keys[2])
    value = network.value_network.init(keys[3])
    return normalizer, policy, value


def write_checkpoint(path: Path, profile: str) -> Path:
    spec = profile_spec(profile)
    env = export_policy.build_env(spec)
    size = obs_size(env)
    network = make_network(spec["network"], size, env.action_size)
    brax_model.save_params(str(path), random_params(network, size, seed=1))
    write_sidecar(path, spec, size, env.action_size)
    return path


@pytest.fixture(scope="module")
def deploy_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_checkpoint(tmp_path_factory.mktemp("deploy") / "run.pkl", "deploy")


@pytest.fixture(scope="module")
def exported(deploy_checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    out = deploy_checkpoint.with_name("run_deploy")
    export_policy.main(["--checkpoint", str(deploy_checkpoint)])
    return out, json.loads((out / "policy_meta.json").read_text())


def test_export_writes_both_files_to_default_dir(
    exported: tuple[Path, dict[str, Any]],
) -> None:
    out, meta = exported
    assert (out / "policy.npz").exists()
    assert meta["run_name"] == "run"
    assert meta["schema_version"] == 2


def test_exported_contract_matches_upstream(
    exported: tuple[Path, dict[str, Any]],
) -> None:
    _, meta = exported
    ignore = {"run_name", "checkpoint"}
    upstream = {k: v for k, v in SNAPSHOT["contract"].items() if k not in ignore}
    assert {k: v for k, v in meta.items() if k not in ignore} == upstream


def test_runtime_loads_export_and_steps_within_bounds(
    exported: tuple[Path, dict[str, Any]],
) -> None:
    out, meta = exported
    policy = WojtekPolicy(out / "policy.npz")
    home = np.asarray(meta["home_ctrl"], np.float32)
    targets = policy.step(np.zeros(3), [0.0, 0.0, -1.0], home, np.zeros(12), [0, 0, 0])
    assert targets.shape == (12,)
    assert np.all(targets >= np.asarray(meta["target_low"]) - 1e-6)
    assert np.all(targets <= np.asarray(meta["target_high"]) + 1e-6)


def test_export_carries_normalizer(
    deploy_checkpoint: Path, exported: tuple[Path, dict[str, Any]]
) -> None:
    out, _ = exported
    _, params = load_run(deploy_checkpoint)
    npz = np.load(out / "policy.npz")
    np.testing.assert_array_equal(npz["norm_mean"], params[0].mean["state"])
    np.testing.assert_array_equal(npz["norm_std"], params[0].std["state"])


def test_numpy_policy_matches_brax_on_export(
    deploy_checkpoint: Path, exported: tuple[Path, dict[str, Any]]
) -> None:
    spec, params = load_run(deploy_checkpoint)
    size = ObsSize(spec["obs_size"]["actor"], spec["obs_size"]["critic"])
    network = make_network(spec["network"], size, spec["action_size"])
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    obs = np.linspace(-1.0, 1.0, size.actor, dtype=np.float32)
    ref, _ = policy(
        {"state": obs, "privileged_state": np.zeros(size.critic, np.float32)},
        jax.random.key(0),
    )
    npz = dict(np.load(exported[0] / "policy.npz"))
    got = export_policy.numpy_policy(npz, obs, spec["action_size"])
    np.testing.assert_allclose(got, np.asarray(ref), atol=1e-5)


def test_benchmark_checkpoint_is_refused(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "bench.pkl", "benchmark")
    with pytest.raises(ValueError, match=r"phase.*SiLU.*--profile deploy"):
        export_policy.main(["--checkpoint", str(checkpoint)])
    assert not (tmp_path / "bench_deploy").exists()


def test_missing_sidecar_names_backfill_script(tmp_path: Path) -> None:
    checkpoint = tmp_path / "old.pkl"
    checkpoint.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="write_sidecar.py"):
        load_run(checkpoint)


def test_sidecar_from_other_run_is_refused(
    deploy_checkpoint: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "mixed.pkl"
    checkpoint.write_bytes(deploy_checkpoint.read_bytes())
    sidecar = json.loads(sidecar_path(deploy_checkpoint).read_text())
    sidecar["obs_size"]["actor"] += 1
    sidecar_path(checkpoint).write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="different runs"):
        load_run(checkpoint)
