"""Paper figures from wandb histories and rendered levels
(`conf/paper_figures.yaml`)."""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, PaperFiguresConfig, to_dataclass
from paper.figures import figures


@hydra.main(
    config_path=CONF_DIR, config_name="paper_figures", version_base="1.3"
)
def main(dict_cfg: DictConfig) -> None:
    figures(to_dataclass(dict_cfg, PaperFiguresConfig))


if __name__ == "__main__":
    main()
