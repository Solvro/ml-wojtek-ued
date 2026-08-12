from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class EnvironmentParams:
    name: str = "joystick"
    envs_num: int = 4
    steps_in_episode: int = 200
    seed: int = 42


@dataclass
class ModelParams:
    dense_dim: int = 128
    activation: str = "tanh"


@dataclass
class TrainingParams:
    iterations: int = 1000
    epochs: int = 10
    minibatches: int = 32
    learning_rate: float = 3e-4
    end_learning_rate: float = 1e-6
    gamma: float = 0.95
    lam: float = 0.95
    epsilon: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    artifacts_dir: str = "artifacts"
    save_every: int = 1000
    eval_every: int = 100
    max_to_keep: int = 5


@dataclass
class EvaluationParams:
    envs_num: int = 5
    steps: int = 200


@dataclass
class WandbParams:
    enable: bool = False
    entity: str = ""
    project: str = ""
    name: str | None = None
    tags: list[str] | None = None


@dataclass
class Config:
    environment_params: EnvironmentParams = field(
        default_factory=EnvironmentParams,
    )
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    evaluation_params: EvaluationParams = field(
        default_factory=EvaluationParams,
    )
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
