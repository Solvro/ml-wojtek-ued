from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from config.hfield_config import HFieldConfig
from config.ppo_config import WandbParams
from utils.checkpoint import Algo


@dataclass
class EvalConfig:
    algo: Algo = MISSING
    run: str | None = None
    step: int | None = None
    artifacts_dir: str = "artifacts"
    envs_num: int = 4096  # H100-scale, matches accel_top_k's num_lanes*envs_per_lane
    steps: int = 200
    seed: int = 0
    wandb_params: WandbParams = field(default_factory=WandbParams)


@dataclass
class EvalLevelsConfig:
    algo: Algo = MISSING
    run: str | None = None
    step: int | None = None
    artifacts_dir: str = "artifacts"
    num_levels: int = 10000
    episodes_per_level: int = 8
    # levels per jit call, at most the arena's tiles (hfield_params.tiles**2);
    # x episodes_per_level = 2048 envs
    chunk_size: int = 256
    steps: int = 400
    # success = no fall and at least this far (m) from spawn at the end
    min_travel: float = 1.0
    cvar_alpha: float = 0.25
    seed: int = 0
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)
cs.store(name="eval_levels_config", node=EvalLevelsConfig)
