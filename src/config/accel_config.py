from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from config.hfield_config import HFieldConfig
from config.ppo_config import (
    EnvironmentParams,
    EvaluationParams,
    ModelParams,
    TrainingParams,
    WandbParams,
)
from utils.checkpoint import Algo


@dataclass
class AccelParams:
    buffer_size: int = 100  # K: level buffer capacity
    initial_fill_ratio: float = 0.9
    replay_prob: float = 0.5
    num_lanes: int = 64
    envs_per_lane: int = 64
    top_k: int = 16
    max_iterations: int = 10000  # hard cap on the while-not-converged loop
    min_iterations: int = 1000  # curation warmup before the check arms
    terrain_coarse_n_start: int = 3  # coarse_n at the flattest curriculum stage
    terrain_coarse_n_stages: int = 4  # discrete steps up to hfield_params.coarse_n
    plateau_window: int = 100  # iterations averaged into the smoothed regret
    plateau_patience: int = 200  # iterations without improvement to converge
    plateau_rel_tol: float = 0.01


@dataclass
class AccelConfig:
    environment_params: EnvironmentParams = field(
        default_factory=EnvironmentParams,
    )
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    evaluation_params: EvaluationParams = field(
        default_factory=EvaluationParams,
    )
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    accel_params: AccelParams = field(default_factory=AccelParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)
    warm_start_algo: Algo | None = None
    warm_start_run: str | None = None


cs = ConfigStore.instance()
cs.store(name="accel_config", node=AccelConfig)
