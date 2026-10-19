"""Throughput profile of a training iteration (`conf/profile.yaml`)."""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, TrainConfig, to_dataclass
from diagnostics.throughput import profile


@hydra.main(config_path=CONF_DIR, config_name="profile", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    profile(to_dataclass(dict_cfg, TrainConfig))


if __name__ == "__main__":
    main()
