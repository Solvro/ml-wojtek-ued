"""Import a w01-tek `train.py` checkpoint as a `brax_ued.py` deploy checkpoint.

    --checkpoint <run>/checkpoints/<step>   brax ppo_checkpoint (orbax) +
                                            ppo_network_config.json; the
                                            run's run.json two levels up
    --out <name>.pkl                        params + sidecar <name>.json

Remaining flags are `brax_ued.py` flags for the run the import stands in for
(`--profile deploy` is required); the sidecar is that run's spec plus
`imported_from`. Refuses, before writing, a checkpoint whose CONSUMED env
keys, actor or critic observation size, action size, network or parameter
shapes differ from the deploy profile's, or whose normalizer keys differ.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
from brax.io import model as brax_model
from brax.training.agents.ppo import checkpoint as ppo_checkpoint

from deploy.contract import CONSUMED_KEYS
from scripts.brax_ued import (
    load_run,
    make_network,
    parse_args,
    run_spec,
    write_sidecar,
)
from scripts.export_policy import build_env
from utils.env import ObsSize

NETWORK_KEYS = (
    "policy_hidden_layer_sizes",
    "value_hidden_layer_sizes",
    "activation",
    "distribution_type",
    "policy_obs_key",
    "value_obs_key",
)
NORMALIZER_KEYS = {"state", "privileged_state"}


def plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def shapes(tree: Any) -> Any:
    return jax.tree.map(lambda x: tuple(np.shape(x)), tree)


def incompatibilities(
    spec: dict[str, Any],
    upstream_env: dict[str, Any],
    net_cfg: dict[str, Any],
    params: Any,
) -> list[str]:
    """Every way the upstream checkpoint differs from the deploy profile."""
    env = build_env(spec)
    ours = plain(env._config.to_dict())
    problems = [
        f"env key {key!r}: upstream {upstream_env.get(key)!r}, ours {ours[key]!r}"
        for key in sorted(CONSUMED_KEYS)
        if plain(upstream_env.get(key)) != ours[key]
    ]
    obs = env.reset(jax.random.key(0)).obs
    size = ObsSize(obs["state"].shape[-1], obs["privileged_state"].shape[-1])
    up_size = {k: v["shape"][0] for k, v in net_cfg["observation_size"].items()}
    if up_size.get("state") != size.actor:
        problems.append(
            f"actor obs: upstream {up_size.get('state')}, ours {size.actor}"
        )
    if up_size.get("privileged_state") != size.critic:
        problems.append(
            f"critic obs: upstream {up_size.get('privileged_state')}, ours "
            f"{size.critic} (the value net cannot be carried over; re-initialising "
            "it is a decision to make, not a default)"
        )
    if net_cfg["action_size"] != env.action_size:
        problems.append(
            f"action size: upstream {net_cfg['action_size']}, ours {env.action_size}"
        )
    kwargs = net_cfg["network_factory_kwargs"]
    for key in NETWORK_KEYS:
        if plain(kwargs.get(key)) != plain(spec["network"][key]):
            problems.append(
                f"network {key}: upstream {kwargs.get(key)!r}, ours "
                f"{spec['network'][key]!r}"
            )
    normalizer = params[0]
    if set(normalizer.mean) != NORMALIZER_KEYS:
        problems.append(
            f"normalizer keys: upstream {sorted(normalizer.mean)}, ours "
            f"{sorted(NORMALIZER_KEYS)}"
        )
    if problems:
        return problems
    network = make_network(spec["network"], size, env.action_size)
    key = jax.random.key(0)
    for name, net, got in (
        ("policy", network.policy_network, params[1]),
        ("value", network.value_network, params[2]),
    ):
        want = shapes(net.init(key))
        if shapes(got) != want:
            problems.append(f"{name} params: upstream {shapes(got)}, ours {want}")
    return problems


def import_checkpoint(
    checkpoint: Path, out: Path, train_flags: list[str]
) -> dict[str, Any]:
    run_json = checkpoint.parents[1] / "run.json"
    config = checkpoint / "ppo_network_config.json"
    for path in (run_json, config):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: --checkpoint must be a w01-tek "
                "runs/<run>/checkpoints/<step> directory"
            )
    spec = run_spec(parse_args(train_flags))
    if spec["profile"] != "deploy":
        raise ValueError(
            "an upstream checkpoint imports as a --profile deploy run; pass "
            "--profile deploy"
        )
    run = json.loads(run_json.read_text())
    net_cfg = json.loads(config.read_text())
    params = ppo_checkpoint.load(str(checkpoint))
    problems = incompatibilities(spec, run["env_config"], net_cfg, params)
    if problems:
        raise ValueError(
            f"{checkpoint} does not match the deploy profile:\n  - "
            + "\n  - ".join(problems)
            + "\nImport a checkpoint trained with upstream's terrain_blind_v4_4d "
            "(the snapshot in tests/deploy) or update the deploy profile first"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    brax_model.save_params(str(out), tuple(params))
    spec["imported_from"] = {
        "checkpoint": str(checkpoint.resolve()),
        "run_name": run["run_name"],
        "env_steps": int(checkpoint.name),
    }
    size = ObsSize(
        net_cfg["observation_size"]["state"]["shape"][0],
        net_cfg["observation_size"]["privileged_state"]["shape"][0],
    )
    write_sidecar(out, spec, size, net_cfg["action_size"])
    spec, _ = load_run(out)
    print(f"wrote {out} and its sidecar from {run['run_name']} @ {checkpoint.name}")
    return spec


def main(argv: list[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="w01-tek checkpoint step dir")
    p.add_argument("--out", required=True, help="brax_ued .pkl to write")
    known, train_flags = p.parse_known_args(argv)
    return import_checkpoint(Path(known.checkpoint), Path(known.out), train_flags)


if __name__ == "__main__":
    main()
