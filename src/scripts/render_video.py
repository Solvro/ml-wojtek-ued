"""mp4 clips of a training checkpoint on generated levels
(docs/ued-accel.md#render-video). Needs MUJOCO_GL=cgl on macOS,
egl/osmesa headless.

Configured by Hydra: `conf/render.yaml`, typed by `config.schema.RenderConfig`.
"""

import hydra
from omegaconf import DictConfig

from config.schema import CONF_DIR, RenderConfig, to_dataclass
from rendering.video import render


@hydra.main(config_path=CONF_DIR, config_name="render", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    render(to_dataclass(dict_cfg, RenderConfig))


if __name__ == "__main__":
    main()
