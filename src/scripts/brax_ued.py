"""Train a brax PPO student under a UED curriculum (`conf/train.yaml`)."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from config.schema import CONF_DIR, TrainConfig, to_dataclass
from training.train import train


@hydra.main(config_path=CONF_DIR, config_name="train", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    mode_name = HydraConfig.get().runtime.choices["mode"]
    train(to_dataclass(dict_cfg, TrainConfig), mode_name)


if __name__ == "__main__":
    main()
