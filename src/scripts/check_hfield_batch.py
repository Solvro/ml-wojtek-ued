"""Check that every env stands on its own arena tile, per backend
(`conf/check_hfield.yaml`)."""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, CheckHFieldConfig, to_dataclass
from diagnostics.arena_check import check


@hydra.main(config_path=CONF_DIR, config_name="check_hfield", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    check(to_dataclass(dict_cfg, CheckHFieldConfig))


if __name__ == "__main__":
    main()
