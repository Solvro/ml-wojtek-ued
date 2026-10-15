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


@dataclass
class PpoDrTerrainConfig:
    environment_params: EnvironmentParams = field(default_factory=EnvironmentParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    evaluation_params: EvaluationParams = field(default_factory=EvaluationParams)
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="ppo_dr_terrain_config", node=PpoDrTerrainConfig)
