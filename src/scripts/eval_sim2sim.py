"""A training checkpoint with its physics on CPU MuJoCo variants
(`conf/sim2sim.yaml`)."""

import json
from pathlib import Path

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig

from config.schema import CONF_DIR, Sim2SimConfig, as_dict, to_dataclass
from evaluation.sim2sim import build, evaluate, level_sets


@hydra.main(config_path=CONF_DIR, config_name="sim2sim", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_dataclass(dict_cfg, Sim2SimConfig)
    run = wandb.init(
        entity=cfg.entity or None,
        project=cfg.project,
        name=cfg.name,
        tags=["sim2sim", cfg.robot.name],
        config=as_dict(cfg),
    )
    ctx = build(cfg)
    metrics, raw = evaluate(ctx, level_sets(cfg, ctx.config))
    run.summary.update(metrics)
    run.finish()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / f"{cfg.name}.npz", **raw)


if __name__ == "__main__":
    main()
