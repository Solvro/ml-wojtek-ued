from collections.abc import Callable, Iterator
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from config.schema import CONF_DIR, SCHEMAS, TrainConfig, to_dataclass

Compose = Callable[..., Any]


@pytest.fixture
def compose_cfg() -> Iterator[Compose]:
    """`compose_cfg(name, *overrides)`: conf/<name>.yaml as its schema
    dataclass, as the entry point's `@hydra.main` would see it."""

    def run(name: str, *overrides: str) -> Any:
        schema = TrainConfig if name == "profile" else SCHEMAS[name]
        return to_dataclass(compose(name, overrides=list(overrides)), schema)

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        yield run
