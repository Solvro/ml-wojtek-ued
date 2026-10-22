"""IQM, stratified bootstrap CIs and probability of improvement from final
evals (`conf/paper_stats.yaml`)."""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, PaperStatsConfig, to_dataclass
from paper.stats import stats


@hydra.main(config_path=CONF_DIR, config_name="paper_stats", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    stats(to_dataclass(dict_cfg, PaperStatsConfig))


if __name__ == "__main__":
    main()
