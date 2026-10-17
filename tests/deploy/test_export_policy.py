import dataclasses
import importlib.util
import json
import os
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
    UPSTREAM_RUNS,
    load_run,
    make_network,
    parse_args,
    run_spec,
    sidecar_path,
    write_sidecar,
)
from utils.env import ObsSize

REPO = Path(__file__).resolve().parents[2]
# the robot's runtime in a w01-tek checkout (read-only); W01_TEK overrides
UPSTREAM_RUNTIME = (
    Path(os.environ.get("W01_TEK", REPO.parent / "w01-tek"))
    / "ros/src/wojtek_policy/wojtek_policy/policy.py"
)


def snapshot(profile: str) -> dict[str, Any]:
    path = Path(__file__).parent / f"upstream_{profile}.json"
    return json.loads(path.read_text())


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


@pytest.fixture(scope="module", params=list(UPSTREAM_RUNS))
def deploy_checkpoint(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    profile = request.param
    return write_checkpoint(tmp_path_factory.mktemp(profile) / "run.pkl", profile)


@pytest.fixture(scope="module")
def exported(deploy_checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    out = deploy_checkpoint.with_name("run_deploy")
    export_policy.main(["--checkpoint", str(deploy_checkpoint)])
    return out, json.loads((out / "policy_meta.json").read_text())


def checkpoint_profile(checkpoint: Path) -> str:
    return json.loads(sidecar_path(checkpoint).read_text())["profile"]


def test_export_writes_both_files_to_default_dir(
    exported: tuple[Path, dict[str, Any]],
) -> None:
    out, meta = exported
    assert (out / "policy.npz").exists()
    assert meta["run_name"] == "run"
    assert meta["schema_version"] == 2


def test_exported_contract_matches_upstream(
    deploy_checkpoint: Path, exported: tuple[Path, dict[str, Any]]
) -> None:
    _, meta = exported
    contract = snapshot(checkpoint_profile(deploy_checkpoint))["contract"]
    ignore = {"run_name", "checkpoint"}
    upstream = {k: v for k, v in contract.items() if k not in ignore}
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
    scale = meta["tau_ff"]["scale"]
    assert policy.last_tau_ff.shape == (12,)
    assert np.all(np.abs(policy.last_tau_ff) <= scale)


def test_tau_ff_head_reaches_the_runtime(
    deploy_checkpoint: Path, exported: tuple[Path, dict[str, Any]]
) -> None:
    out, meta = exported
    if not meta["tau_ff"]["enable"]:
        pytest.skip("this profile's contract has no tau_ff head")
    spec, _ = load_run(deploy_checkpoint)
    assert spec["action_size"] == meta["action_size"] == 24
    policy = WojtekPolicy(out / "policy.npz")
    home = np.asarray(meta["home_ctrl"], np.float32)
    dq = np.linspace(-3.0, 3.0, 12)
    policy.step(np.zeros(3), [0.0, 0.0, -1.0], home + 0.1, dq, [0.3, 0.0, 0.2])
    obs = np.asarray(policy.last_obs, np.float32)
    action = export_policy.numpy_policy(
        dict(np.load(out / "policy.npz")), obs, meta["action_size"]
    )
    np.testing.assert_allclose(
        policy.last_tau_ff, action[12:] * meta["tau_ff"]["scale"], atol=1e-5
    )


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


def upstream_runtime() -> Any:
    if not UPSTREAM_RUNTIME.exists():
        pytest.skip(f"no w01-tek checkout at {UPSTREAM_RUNTIME} (set W01_TEK)")
    spec = importlib.util.spec_from_file_location("upstream_policy", UPSTREAM_RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_runtime_is_upstreams_verbatim() -> None:
    upstream_runtime()
    vendored = (REPO / "src/deploy/runtime.py").read_text().splitlines()
    # the vendored copy only adds its provenance line
    assert vendored[1:] == UPSTREAM_RUNTIME.read_text().splitlines()


def test_upstream_runtime_matches_vendored_on_export(
    exported: tuple[Path, dict[str, Any]],
) -> None:
    out, meta = exported
    theirs = upstream_runtime().WojtekPolicy(out / "policy.npz")
    ours = WojtekPolicy(out / "policy.npz")
    home = np.asarray(meta["home_ctrl"], np.float32)
    rng = np.random.default_rng(0)
    for _ in range(64):
        inputs = (
            rng.uniform(-2, 2, 3),
            rng.uniform(-1, 1, 3),
            home + rng.uniform(-0.4, 0.4, 12),
            rng.uniform(-6, 6, 12),
            rng.uniform(meta["command_low"][:3], meta["command_high"][:3]),
        )
        np.testing.assert_array_equal(ours.step(*inputs), theirs.step(*inputs))
        np.testing.assert_array_equal(ours.last_tau_ff, theirs.last_tau_ff)
