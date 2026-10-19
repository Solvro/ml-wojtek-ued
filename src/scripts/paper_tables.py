"""LaTeX tables and a per-run CSV from final evals (`conf/paper_tables.yaml`)."""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, PaperTablesConfig, to_dataclass
from paper.tables import tables


@hydra.main(config_path=CONF_DIR, config_name="paper_tables", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    tables(to_dataclass(dict_cfg, PaperTablesConfig))


if __name__ == "__main__":
    main()
