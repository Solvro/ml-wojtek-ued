from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from config.hfield_config import HFieldConfig


@dataclass
class ArenaConfig:
    hfield_params: HFieldConfig = field(default_factory=HFieldConfig)
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="arena_config", node=ArenaConfig)
