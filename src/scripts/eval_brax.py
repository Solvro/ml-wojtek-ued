"""Final evaluation of a training checkpoint (docs/ued-accel.md#eval-brax).

Configured by Hydra: `conf/eval.yaml`, typed by `config.schema.EvalConfig`.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig

from config.schema import CONF_DIR, EvalConfig, as_dict, to_dataclass
from evaluation.final_eval import build, evaluate


@hydra.main(config_path=CONF_DIR, config_name="eval", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_dataclass(dict_cfg, EvalConfig)
    run = wandb.init(
        entity=cfg.entity or None,
        project=cfg.project,
        name=cfg.name,
        tags=["brax-ued", "final-eval", cfg.robot.name, *cfg.tags],
        config=as_dict(cfg),
    )
    metrics, raw = evaluate(build(cfg))
    run.summary.update(metrics)
    run.finish()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / f"{cfg.name}.npz", **raw)


if __name__ == "__main__":
    main()
