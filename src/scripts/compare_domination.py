"""Level-by-level domination comparison of eval_brax.py dumps
(`conf/domination.yaml`)."""

import dataclasses
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, DominationConfig, to_dataclass
from evaluation.domination import compare
from paper.domination import plot_matrices, plot_pairs


@hydra.main(config_path=CONF_DIR, config_name="domination", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_dataclass(dict_cfg, DominationConfig)
    report = compare(cfg)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "domination.json"
    path.write_text(json.dumps(dataclasses.asdict(report), indent=2))
    written = [path, *plot_pairs(report, out), plot_matrices(report, out)]
    print("\n".join(f"wrote {p}" for p in written))


if __name__ == "__main__":
    main()
