from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue

from config.hfield_config import HFieldConfig

# conf/ sits beside src/, like the assets wojtek_env.paths resolves
CONF_DIR = str(Path(__file__).resolve().parents[2] / "conf")


class Robot(StrEnum):
    wojtek = "wojtek"
    go1 = "go1"


class Backend(StrEnum):
    auto = "auto"
    jax = "jax"
    warp = "warp"


class Curriculum(StrEnum):
    dr = "dr"
    dr_easy = "dr_easy"
    dr_mixed = "dr_mixed"
    accel = "accel"
    plr = "plr"
    paired = "paired"


class Mutation(StrEnum):
    local = "local"
    grow = "grow"


class Score(StrEnum):
    pvl = "pvl"
    learnability = "learnability"


class Generator(StrEnum):
    easy = "easy"
    mixed = "mixed"


class LevelSpaceKind(StrEnum):
    grid = "grid"
    param = "param"


@dataclass
class RobotConfig:
    name: Robot = MISSING


@dataclass
class LevelSpaceConfig:
    kind: LevelSpaceKind = MISSING
    n_edits_by_mutation: dict[str, int] = MISSING
    n_edits: int = MISSING
    edit_scale: float = MISSING


@dataclass
class ModeConfig:
    curriculum: Curriculum = MISSING
    replay_prob: float = MISSING
    buffer_size: int = MISSING
    generator_scale: float = MISSING
    mutation: Mutation = MISSING
    grow_max: float = MISSING
    score: Score = MISSING
    generator: Generator = MISSING
    latent_dim: int = MISSING
    gen_learning_rate: float = MISSING
    gen_entropy_coef: float = MISSING


@dataclass
class WandbConfig:
    name: str = MISSING
    entity: str = MISSING
    project: str = MISSING


@dataclass
class TrainConfig(WandbConfig):
    robot: RobotConfig = field(default_factory=RobotConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    level_space: LevelSpaceConfig = field(default_factory=LevelSpaceConfig)
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    backend: Backend = MISSING
    seed: int = MISSING
    num_timesteps: int = MISSING
    num_lanes: int = MISSING
    envs_per_lane: int = MISSING
    level_steps: int = MISSING
    eval_every: int = MISSING
    eval_levels: int = MISSING
    min_travel: float = MISSING
    checkpoint_dir: str | None = MISSING
    tags: list[str] = MISSING


@dataclass
class EvalConfig(WandbConfig):
    robot: RobotConfig = field(default_factory=RobotConfig)
    level_space: LevelSpaceConfig = field(default_factory=LevelSpaceConfig)
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    backend: Backend = MISSING
    checkpoint: str = MISSING
    out_dir: str = MISSING
    num_levels: int = MISSING
    episodes: int = MISSING
    suite_instances: int = MISSING
    chunk_envs: int = MISSING
    level_steps: int = MISSING
    min_travel: float = MISSING
    alphas: list[float] = MISSING
    seed: int = MISSING
    tags: list[str] = MISSING


@dataclass
class Sim2SimConfig(WandbConfig):
    robot: RobotConfig = field(default_factory=RobotConfig)
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    checkpoint: str = MISSING
    out_dir: str = MISSING
    num_levels: int = MISSING
    episodes: int = MISSING
    suite_instances: int = MISSING
    chunk_envs: int = MISSING
    level_steps: int = MISSING
    min_travel: float = MISSING
    seed: int = MISSING


@dataclass
class RenderConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    level_space: LevelSpaceConfig = field(default_factory=LevelSpaceConfig)
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    backend: Backend = MISSING
    checkpoint: str = MISSING
    out_dir: str = MISSING
    levels: list[str] = MISSING
    seconds: float = MISSING
    fps: int = MISSING
    width: int = MISSING
    height: int = MISSING
    command: list[float] = MISSING
    seed: int = MISSING


@dataclass
class ArenaBuildConfig:
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    seed: int = MISSING


@dataclass
class CheckHFieldConfig:
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    backend: Backend = MISSING
    pit: float = MISSING
    tiles: int = MISSING
    envs_per_tile: int = MISSING
    steps: int = MISSING
    compare: bool = MISSING
    tol: float = MISSING
    bench_envs: int = MISSING


@dataclass
class PaperTablesConfig:
    prefixes: dict[Robot, str | None] = MISSING
    eval_dir: str | None = MISSING
    out: str = MISSING
    train_summaries: str | None = MISSING
    dump_wandb: str | None = MISSING


@dataclass
class PaperFiguresConfig:
    prefixes: dict[Robot, str | None] = MISSING
    arena: HFieldConfig = field(default_factory=HFieldConfig)
    dump_history: str | None = MISSING
    history: str | None = MISSING
    terrains: bool = MISSING
    out: str = MISSING


SCHEMAS: dict[str, type] = {
    "train": TrainConfig,
    "eval": EvalConfig,
    "sim2sim": Sim2SimConfig,
    "render": RenderConfig,
    "arena": ArenaBuildConfig,
    "check_hfield": CheckHFieldConfig,
    "paper_tables": PaperTablesConfig,
    "paper_figures": PaperFiguresConfig,
}

for _name, _schema in SCHEMAS.items():
    ConfigStore.instance().store(name=f"{_name}_schema", node=_schema)

T = TypeVar("T")


def _run_name(*, _root_: DictConfig) -> str:
    try:
        return str(_root_.name)
    except MissingMandatoryValue:
        raise ValueError(
            "name is required: it names the wandb run and the "
            "artifacts/hydra/<name> directory; pass name=<run name>"
        ) from None


OmegaConf.register_new_resolver("run_name", _run_name, replace=True)


def to_dataclass(cfg: DictConfig, schema: type[T]) -> T:
    """The composed config as its schema dataclass; fails on an unset `???`."""
    obj: Any = OmegaConf.to_object(cfg)
    if not isinstance(obj, schema):
        raise TypeError(f"composed config is {type(obj).__name__}, not {schema}")
    return obj


def as_dict(cfg: Any) -> dict[str, Any]:
    """A schema dataclass as plain values (enums as strings), for wandb."""
    out: Any = OmegaConf.to_container(OmegaConf.structured(cfg), enum_to_str=True)
    return out
