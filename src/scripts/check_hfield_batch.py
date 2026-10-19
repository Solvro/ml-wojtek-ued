"""Check that every env of a batch stands on its own tile of the shared
arena; `compare=true` runs jax and warp, `bench_envs=N` times env steps
(docs/environment.md#tiled-arena). Needs the `accel` arena built from the
same `arena` config.

Configured by Hydra: `conf/check_hfield.yaml`, typed by
`config.schema.CheckHFieldConfig`.
"""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, CheckHFieldConfig, to_dataclass
from diagnostics.arena_check import check


@hydra.main(config_path=CONF_DIR, config_name="check_hfield", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    check(to_dataclass(dict_cfg, CheckHFieldConfig))


if __name__ == "__main__":
    main()
