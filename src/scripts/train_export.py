"""Train a policy (`conf/train.yaml`, e.g. `policy=<name>`) and export it."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from config.schema import CONF_DIR, TrainConfig, to_dataclass
from export.run import export_policy
from training.train import train


@hydra.main(config_path=CONF_DIR, config_name="train", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_dataclass(dict_cfg, TrainConfig)
    if not cfg.checkpoint_dir:
        raise ValueError(
            "train-export needs checkpoint_dir: set it in the policy file "
            "or pass checkpoint_dir=<dir>"
        )
    train(cfg, HydraConfig.get().runtime.choices["mode"])
    checkpoint = Path(cfg.checkpoint_dir) / f"{cfg.name}.pkl"
    out_dir, _ = export_policy(cfg, checkpoint, Path(cfg.export_dir))
    print(f"exported {checkpoint} to {out_dir}")


if __name__ == "__main__":
    main()
