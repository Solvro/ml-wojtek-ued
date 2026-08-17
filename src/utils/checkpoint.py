import json
import time
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

import equinox as eqx
import orbax.checkpoint as ocp


class Algo(str, Enum):
    ppo = "ppo"
    ppo_dr = "ppo_dr"
    accel = "accel"
    accel_top_k = "accel_top_k"
    paired = "paired"
    paired_evo = "paired_evo"


class RunMeta(TypedDict):
    algo: str
    created_at: float
    env_task: str
    env_overrides: dict[str, Any] | None
    env_seed: int
    dense_dim: int
    activation: str
    action_dim: int
    actor_obs_dim: int
    critic_obs_dim: int
    action_high: float


def resolve_run_name(configured_name: str | None) -> str:
    return configured_name or f"run-{int(time.time())}"


def run_dir(artifacts_dir: str | Path, algo: str, name: str) -> Path:
    return Path(artifacts_dir).resolve() / algo / name


def latest_run_dir(artifacts_dir: str | Path, algo: str) -> Path:
    algo_dir = Path(artifacts_dir).resolve() / algo
    runs = [d for d in algo_dir.iterdir() if d.is_dir()] if algo_dir.is_dir() else []
    if not runs:
        raise FileNotFoundError(
            f"no runs found for algo '{algo}' under {algo_dir}. "
            "train a model first, or pass --run <name> explicitly"
        )
    return max(runs, key=lambda d: d.stat().st_mtime)


def make_checkpoint_manager(
    directory: Path, max_to_keep: int, save_interval_steps: int
) -> ocp.CheckpointManager:
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep, save_interval_steps=save_interval_steps
    )
    return ocp.CheckpointManager(
        directory, item_names=("model", "opt_state"), options=options
    )


def open_checkpoint_manager(directory: Path) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(directory, item_names=("model", "opt_state"))


def save_meta(directory: Path, meta: RunMeta) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def load_meta(directory: Path) -> RunMeta:
    path = directory / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no meta.json in {directory}; this run predates the checkpoint "
            "metadata convention (see utils/checkpoint.py) and can't be "
            "restored by scripts/eval.py"
        )
    return json.loads(path.read_text())


def save_checkpoint(
    ckptr: ocp.CheckpointManager, step: int, model: Any, opt_state: Any
) -> bool:
    array_model, _ = eqx.partition(model, eqx.is_array)
    return ckptr.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(array_model),
            opt_state=ocp.args.StandardSave(opt_state),
        ),
    )


def restore_checkpoint(
    ckptr: ocp.CheckpointManager,
    step: int | None,
    template_model: Any,
    template_opt_state: Any | None = None,
) -> tuple[Any, Any | None]:
    # template_opt_state=None (eval's case) restores the model only, skipping
    # the optimizer item entirely rather than requiring a throwaway optimizer.
    array_template, static = eqx.partition(template_model, eqx.is_array)
    step = ckptr.latest_step() if step is None else step
    if step is None:
        raise FileNotFoundError(f"no checkpoints found in {ckptr.directory}")

    items: dict[str, Any] = {"model": ocp.args.StandardRestore(array_template)}
    if template_opt_state is not None:
        items["opt_state"] = ocp.args.StandardRestore(template_opt_state)

    restored = ckptr.restore(step, args=ocp.args.Composite(**items))
    model = eqx.combine(restored.model, static)
    opt_state = restored.opt_state if template_opt_state is not None else None
    return model, opt_state
