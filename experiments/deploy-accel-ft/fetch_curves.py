"""In-training eval curves of the dft2 runs, from wandb, as JSON.

    uv run python experiments/deploy-accel-ft/fetch_curves.py \
        --out experiments/deploy-accel-ft/cache/curves.json
"""

import argparse
import json
from pathlib import Path

import wandb

PROJECT = "dominikgalus9-wroc-aw-university-of-science-and-technology/wojtyla"
RUNS = r"^dft2_deploy_(accel|accel_lr1e-4|dr)_seed[0-9]$"
KEYS = [
    "env_steps",
    "eval/d1.0/success_rate",
    "eval/d1.0/cmd_success_rate",
    "eval/d0.25/success_rate",
    "eval/d1.0/fall_rate",
]
TRAIN_KEYS = ["env_steps", "diverged_per_step", "level/genome_dist", "train_success"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    api = wandb.Api(timeout=120)
    out = {}
    for run in api.runs(PROJECT, filters={"display_name": {"$regex": RUNS}}):
        out[run.name] = {
            "state": run.state,
            "eval": list(run.scan_history(keys=KEYS)),
            "train": list(run.scan_history(keys=TRAIN_KEYS)),
            "elapsed_h": run.summary.get("final/elapsed_h"),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(out)} runs")


if __name__ == "__main__":
    main()
