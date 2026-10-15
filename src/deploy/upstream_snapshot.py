"""Snapshot what w01-tek resolves for the deploy experiment, with its own code.

Composes upstream's Hydra config for `+experiment=terrain_blind_v4_4d`
(read-only), then records: the full resolved env config (`run.json`'s
`env_config` minus the per-host `sim.num_envs`), the `dr` block, the PPO
network factory kwargs (`train.py build_ppo_params` + the network group, with
brax's defaults for activation / distribution made explicit), and the
contract upstream's exporter builds. tests/deploy compares `--profile deploy`
against it.

    uv run python -m deploy.upstream_snapshot --w01-tek ../w01-tek
"""

import argparse
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

EXPERIMENT = "terrain_blind_v4_4d"
OUT = Path(__file__).resolve().parents[2] / f"tests/deploy/upstream_{EXPERIMENT}.json"


def network_kwargs(cfg: Any) -> dict[str, Any]:
    from brax.training.agents.ppo import networks as ppo_networks
    from wojtek_rl.train import _apply_ppo_overrides, build_ppo_params

    params = build_ppo_params({}, smoke=False)
    network = OmegaConf.to_container(cfg.network, resolve=True) or {}
    _apply_ppo_overrides(params.network_factory, network)
    kwargs = params.network_factory.to_dict()
    defaults = inspect.signature(ppo_networks.make_ppo_networks).parameters
    activation = kwargs.get("activation", defaults["activation"].default)
    return {
        **{k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()},
        "activation": activation.__name__,
        "distribution_type": kwargs.get(
            "distribution_type", defaults["distribution_type"].default
        ),
    }


def snapshot(w01_tek: Path) -> dict[str, Any]:
    from wojtek_rl.deploy_contract import build_contract
    from wojtek_rl.export_policy import build_env
    from wojtek_rl.registry import TASKS, _apply_overrides

    conf = w01_tek / "training/wojtek_rl/conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf)):
        cfg = compose(config_name="config", overrides=[f"+experiment={EXPERIMENT}"])
    task = str(cfg.task.name)
    overrides = OmegaConf.to_container(cfg.task.env, resolve=True)
    env_config = TASKS[task][1]()
    _apply_overrides(env_config, overrides)
    run = {"run_name": EXPERIMENT, "task": task, "env_config": env_config.to_dict()}
    env_dict = json.loads(json.dumps(run["env_config"]))
    del env_dict["sim"]["num_envs"]
    commit = subprocess.run(
        ["git", "-C", str(w01_tek), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "w01_tek_commit": commit,
        "experiment": EXPERIMENT,
        "task": task,
        "env_config": env_dict,
        "dr": OmegaConf.to_container(cfg.dr, resolve=True),
        "network": network_kwargs(cfg),
        "contract": build_contract(build_env(run), run),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--w01-tek", type=Path, required=True, help="w01-tek checkout")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    # keep the upstream checkout untouched
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.w01_tek.resolve() / "training"))
    record = snapshot(args.w01_tek.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out} (w01-tek {record['w01_tek_commit']})")


if __name__ == "__main__":
    main()
