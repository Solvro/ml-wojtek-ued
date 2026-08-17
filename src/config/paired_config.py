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
class PairedParams:
    num_lanes: int = 32  # generated levels scored per iteration
    envs_per_lane: int = 32  # episodes rolled out per level (per student)

    latent_dim: int = 16  # dimension of the z seed fed to the generator
    gen_dense_dim: int = 128
    gen_activation: str = "tanh"
    gen_learning_rate: float = 1e-4
    gen_epochs: int = 5  # single-step PPO epochs per iteration
    gen_epsilon: float = 0.2  # generator PPO clip range
    gen_entropy_coef: float = 0.1  # HiEnt: high entropy bonus on the teacher
    gen_max_grad_norm: float = 0.5

    student_entropy_coef: float = 0.05  # HiEnt bonus on both students

    bc_mode: str = "none"  # "none" | "uni" (UniBC) | "bi" (BiBC)
    bc_coef: float = 0.5  # KL distillation weight
    kl_update_step: int = 1  # apply BC every N iterations (periodic distillation)

    max_iterations: int = 10000  # hard cap on the while-not-converged loop
    min_iterations: int = 1000  # warmup before the plateau check arms
    plateau_window: int = 100
    plateau_patience: int = 200
    plateau_rel_tol: float = 0.01


@dataclass
class PairedConfig:
    environment_params: EnvironmentParams = field(default_factory=EnvironmentParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    evaluation_params: EvaluationParams = field(default_factory=EvaluationParams)
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    paired_params: PairedParams = field(default_factory=PairedParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)


@dataclass
class PairedEvoParams:
    buffer_size: int = 100
    initial_fill_ratio: float = 0.9
    replay_prob: float = 0.5

    student_entropy_coef: float = 0.05  # HiEnt bonus on both students

    bc_mode: str = "none"  # "none" | "uni" (UniBC) | "bi" (BiBC)
    bc_coef: float = 0.5
    kl_update_step: int = 1

    max_iterations: int = 10000
    min_iterations: int = 1000
    plateau_window: int = 100
    plateau_patience: int = 200
    plateau_rel_tol: float = 0.01


@dataclass
class PairedEvoConfig:
    environment_params: EnvironmentParams = field(default_factory=EnvironmentParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    evaluation_params: EvaluationParams = field(default_factory=EvaluationParams)
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    paired_evo_params: PairedEvoParams = field(default_factory=PairedEvoParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="paired_config", node=PairedConfig)
cs.store(name="paired_evo_config", node=PairedEvoConfig)
