"""Export a checkpoint as ONNX with its README (`conf/export.yaml`)."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR
from export.run import export_policy, training_config


@hydra.main(config_path=CONF_DIR, config_name="export", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = training_config(dict_cfg)
    checkpoint = Path(dict_cfg.checkpoint)
    out_dir, _ = export_policy(cfg, checkpoint, Path(cfg.export_dir))
    print(f"exported {checkpoint} to {out_dir}")


if __name__ == "__main__":
    main()
