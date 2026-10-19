"""Paper figures from wandb training histories and rendered levels
(docs/ued-accel.md#paper-figures).

Configured by Hydra: `conf/paper_figures.yaml`, typed by
`config.schema.PaperFiguresConfig`.
"""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, PaperFiguresConfig, to_dataclass
from paper.figures import figures


@hydra.main(config_path=CONF_DIR, config_name="paper_figures", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    figures(to_dataclass(dict_cfg, PaperFiguresConfig))


if __name__ == "__main__":
    main()
