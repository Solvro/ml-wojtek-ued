"""A brax_ued robot-profile .pkl as a w01-tek run dir, for upstream's battery.

    --checkpoint <name>.pkl   brax_ued checkpoint (sidecar beside it)
    --base_run <run dir>      the upstream run it was fine-tuned from
    --out_run <run dir>       written: run.json + checkpoints/<step>

The battery (`./run.sh report --run <dir>`) rebuilds its measurement env
from run.json, so the base's run.json is reused with the new run_name: the
fine-tune shares every CONSUMED key with it (import_upstream checked them),
and the battery measures on upstream's own flat scene either way. The params
go into upstream's checkpoint layout with the base's ppo_network_config.json.
Converting the imported base itself must reproduce the base's own battery.
"""

import argparse
import json
from pathlib import Path

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from ml_collections import config_dict

from scripts.brax_ued import load_run

CONFIG = "ppo_network_config.json"


def latest_checkpoint(run_dir: Path) -> Path:
    steps = sorted(
        (p for p in (run_dir / "checkpoints").iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not steps:
        raise FileNotFoundError(f"no checkpoints/<step> dirs under {run_dir}")
    return steps[-1]


def convert(checkpoint: Path, base_run: Path, out_run: Path) -> Path:
    spec, params = load_run(checkpoint)
    base_ckpt = latest_checkpoint(base_run)
    config = json.loads((base_ckpt / CONFIG).read_text())
    sizes = {k: v["shape"][0] for k, v in config["observation_size"].items()}
    ours = {
        "state": spec["obs_size"]["actor"],
        "privileged_state": spec["obs_size"]["critic"],
    }
    if sizes != ours or config["action_size"] != spec["action_size"]:
        raise ValueError(
            f"{checkpoint} has obs {ours} / action {spec['action_size']}, the "
            f"base run {base_run} obs {sizes} / action {config['action_size']}; "
            "convert a fine-tune of that base"
        )
    run = json.loads((base_run / "run.json").read_text())
    run["run_name"] = out_run.name
    run["converted_from"] = {"checkpoint": str(checkpoint.resolve())}
    out_run.mkdir(parents=True, exist_ok=True)
    (out_run / "run.json").write_text(json.dumps(run, indent=2))
    ppo_checkpoint.save(
        str(out_run / "checkpoints"),
        int(base_ckpt.name),
        tuple(params),
        config_dict.ConfigDict(config),
    )
    return out_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--base_run", required=True, type=Path)
    p.add_argument("--out_run", required=True, type=Path)
    args = p.parse_args()
    out = convert(args.checkpoint, args.base_run, args.out_run)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
