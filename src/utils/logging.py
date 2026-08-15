import json
import logging
import time
from pathlib import Path
from typing import Any


def quiet_dependencies() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("absl", "jax", "orbax"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _to_serializable(obj: Any) -> Any:
    # jax.Array / numpy arrays and scalars aren't JSON-serializable directly.
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class Logger:
    def __init__(self, name: str | None = None, log_dir: str | Path = "logs"):
        name = name or f"run-{int(time.time())}"
        self._path = Path(log_dir) / f"{name}.log"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        record = {"step": step, **data} if step is not None else dict(data)
        with self._path.open("a") as f:
            f.write(json.dumps(record, default=_to_serializable) + "\n")

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        pass

    def finish(self) -> None:
        pass
