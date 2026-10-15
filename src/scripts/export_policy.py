"""Export a `brax_ued.py --profile deploy` checkpoint in w01-tek's robot format.

Port of w01-tek `training/wojtek_rl/export_policy.py` @ e44c1a9. Writes into
`--out` (default `<checkpoint stem>_deploy/` beside the .pkl):

  policy.npz        norm_mean/norm_std + hidden_{i}_kernel/bias MLP layers
  policy_meta.json  the schema-2 deployment contract (deploy/contract.py)

The env is rebuilt from the checkpoint's sidecar on the jax backend, one env,
terrain off (sim and terrain are training-only), and the contract is read off
it. Two validations, as upstream: the numpy forward pass vs the jitted brax
inference (open loop, < 1e-4), and the robot runtime (deploy/runtime.py) on
the written files vs an env-derived reference (closed loop, < 1e-3).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

from deploy.contract import build_contract
from deploy.runtime import WojtekPolicy
from scripts.brax_ued import load_run, spec_network
from utils.env_overrides import merge_overrides
from wojtek_env.envs import registry

# position targets are the first 12 outputs; a tau_ff head doubles the action
TARGET_SIZE = 12
OPEN_LOOP_TOL = 1e-4
# closed loop: last_act and the filter feed each side its own output, so the
# per-step jax-vs-numpy noise accumulates over the 32 steps (upstream's bound)
CLOSED_LOOP_TOL = 1e-3


def numpy_policy(
    params: dict[str, np.ndarray], obs: np.ndarray, action_size: int
) -> np.ndarray:
    x = (obs - params["norm_mean"]) / params["norm_std"]
    i = 0
    while f"hidden_{i}_kernel" in params:
        x = x @ params[f"hidden_{i}_kernel"] + params[f"hidden_{i}_bias"]
        if f"hidden_{i + 1}_kernel" in params:
            x = x * np.exp(-np.logaddexp(0.0, -x))  # SiLU on all but the last
        i += 1
    return np.tanh(x[:action_size])


def build_env(spec: dict[str, Any]) -> Any:
    # sim and terrain are training-only: the jax backend, one env and the flat
    # scene change nothing the contract reads, and need no arena files
    overrides = merge_overrides(
        spec["env_overrides"],
        {"sim": {"backend": "jax", "num_envs": 1}, "terrain": {"enable": False}},
    )
    return registry.make_env(spec["task"], overrides)


def check_deployable(checkpoint: Path, spec: dict[str, Any], env: Any) -> None:
    network = spec["network"]
    problems = []
    if "phase" in env.actor_obs_names:
        problems.append(
            "its actor observes the gait clock ('phase'), which the robot "
            "runtime cannot produce"
        )
    if network["activation"] != "silu":
        problems.append(
            f"its MLP uses {network['activation']!r}, the robot runtime evaluates SiLU"
        )
    if network["distribution_type"] != "tanh_normal":
        problems.append(
            f"its policy head is {network['distribution_type']!r}, the robot "
            "runtime applies tanh to the mean"
        )
    if problems:
        raise ValueError(
            f"{checkpoint} ({spec['profile']} profile) cannot run on the "
            f"robot: {'; '.join(problems)}. Train with `brax_ued.py --profile "
            "deploy`"
        )


def flatten(params: Any) -> dict[str, np.ndarray]:
    normalizer, policy = params[0], params[1]
    export = {
        "norm_mean": np.asarray(normalizer.mean["state"], np.float32),
        "norm_std": np.asarray(normalizer.std["state"], np.float32),
    }
    layers = policy["params"]
    for i, name in enumerate(sorted(layers, key=lambda n: int(n.split("_")[-1]))):
        export[f"hidden_{i}_kernel"] = np.asarray(layers[name]["kernel"], np.float32)
        export[f"hidden_{i}_bias"] = np.asarray(layers[name]["bias"], np.float32)
    return export


def validate_numpy(
    export: dict[str, np.ndarray], inference: Any, sizes: dict[str, int]
) -> float:
    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    worst = 0.0
    for _ in range(32):
        obs = rng.uniform(-2.0, 2.0, sizes["actor"]).astype(np.float32)
        privileged = np.zeros(sizes["critic"], np.float32)
        ref, _ = inference({"state": obs, "privileged_state": privileged}, key)
        got = numpy_policy(export, obs, sizes["action"])
        worst = max(worst, float(np.max(np.abs(np.asarray(ref) - got))))
    if worst >= OPEN_LOOP_TOL:
        raise ValueError(f"numpy forward vs brax inference: max |diff| = {worst}")
    return worst


def validate_runtime(
    env: Any, meta: dict[str, Any], out_dir: Path, inference: Any, priv_size: int
) -> float:
    """The robot runtime on the written files vs the env's own fields."""
    policy = WojtekPolicy(out_dir / "policy.npz")
    home = np.asarray(env._home_ctrl, np.float32)
    anchor = np.asarray(meta["anchor_ctrl"], np.float32)
    scale = np.asarray(meta["action_scale"], np.float32)
    lo = np.asarray(meta["target_low"], np.float32)
    hi = np.asarray(meta["target_high"], np.float32)
    af = float(meta["action_filter"])
    cmd_lo = np.asarray(meta["command_low"], np.float32)
    cmd_hi = np.asarray(meta["command_high"], np.float32)
    fill = np.asarray(meta["command_fill"], np.float32)
    tau_ff = meta["tau_ff"]
    tau_scale = float(tau_ff["scale"])

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    last_act = np.zeros(TARGET_SIZE, np.float32)
    filt = np.zeros(int(meta["action_size"]), np.float32)
    worst = 0.0
    for _ in range(32):
        gyro = rng.uniform(-2, 2, 3).astype(np.float32)
        grav = rng.uniform(-1, 1, 3).astype(np.float32)
        q = (home + rng.uniform(-0.4, 0.4, 12)).astype(np.float32)
        dq = rng.uniform(-6, 6, 12).astype(np.float32)
        cmd3 = rng.uniform(cmd_lo[:3], cmd_hi[:3]).astype(np.float32)
        parts = {
            "gyro": gyro,
            "gravity": grav,
            "joint_pos": q - home,
            "joint_vel": dq,
            # the env's last_act obs is the position half only
            "last_act": last_act,
            "command": np.concatenate([cmd3, fill]),
        }
        obs = np.concatenate([parts[n] for n in env.actor_obs_names])
        privileged = np.zeros(priv_size, np.float32)
        ref_act, _ = inference(
            {"state": obs.astype(np.float32), "privileged_state": privileged}, key
        )
        ref_act = np.asarray(ref_act, np.float32)
        last_act = ref_act[:TARGET_SIZE]
        filt = af * filt + (1.0 - af) * ref_act
        ref_targets = np.clip(anchor + filt[:TARGET_SIZE] * scale, lo, hi)
        got = policy.step(gyro, grav, q, dq, cmd3)
        worst = max(worst, float(np.max(np.abs(ref_targets - got))))
        if tau_ff["enable"]:
            ref_tau = np.clip(filt[TARGET_SIZE:] * tau_scale, -tau_scale, tau_scale)
            # normalized action units, as upstream: raw N*m scales the noise
            tau_diff = np.max(np.abs(ref_tau - policy.last_tau_ff)) / tau_scale
            worst = max(worst, float(tau_diff))
    if worst >= CLOSED_LOOP_TOL:
        raise ValueError(f"deploy runtime vs env reference: max |diff| = {worst}")
    return worst


def export(checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    spec, params = load_run(checkpoint)
    if spec["robot"] != "wojtek":
        raise ValueError(f"{checkpoint} is a {spec['robot']} policy, not Wojtek's")
    env = build_env(spec)
    check_deployable(checkpoint, spec, env)
    run = {
        "run_name": checkpoint.stem,
        "task": spec["task"],
        "env_config": env._config.to_dict(),
    }
    meta = build_contract(env, run, checkpoint=str(checkpoint))
    sizes = {
        "actor": spec["obs_size"]["actor"],
        "critic": spec["obs_size"]["critic"],
        "action": spec["action_size"],
    }
    if (meta["obs_size"], meta["action_size"]) != (sizes["actor"], sizes["action"]):
        raise ValueError(
            f"the env rebuilt from {checkpoint}'s sidecar has obs/action sizes "
            f"{meta['obs_size']}/{meta['action_size']}, the checkpoint "
            f"{sizes['actor']}/{sizes['action']}"
        )
    make_policy = ppo_networks.make_inference_fn(spec_network(spec))
    inference = jax.jit(make_policy(params, deterministic=True))
    flat = flatten(params)
    worst = validate_numpy(flat, inference, sizes)
    print(f"validated numpy vs brax inference: max |diff| = {worst:.2e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # keys are layer names, never savez's allow_pickle
    np.savez(out_dir / "policy.npz", **flat)  # ty: ignore[invalid-argument-type]
    (out_dir / "policy_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    worst = validate_runtime(env, meta, out_dir, inference, sizes["critic"])
    print(f"validated deploy runtime vs env reference: max |diff| = {worst:.2e}")
    print(f"wrote {out_dir}/policy.npz and policy_meta.json (schema 2)")
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="brax_ued.py .pkl, sidecar .json beside it"
    )
    p.add_argument("--out", default=None, help="default: <checkpoint stem>_deploy/")
    args = p.parse_args(argv)
    checkpoint = Path(args.checkpoint)
    out = (
        Path(args.out)
        if args.out
        else checkpoint.with_name(f"{checkpoint.stem}_deploy")
    )
    export(checkpoint, out)


if __name__ == "__main__":
    main()
