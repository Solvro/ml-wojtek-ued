import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

from config.schema import PaperFiguresConfig
from paper.labels import METHODS, ROBOT_LABEL
from paper.runs import ENTITY, PROJECT, run_prefixes
from robots.scale import ROBOT_SCALE

EVAL_KEYS = ("eval/d0.25/success_rate", "eval/d0.5/success_rate")
# magnitude parameters only: the shape ones (wavelength, step width, block
# size) sit at mid-range under every method
LEVEL_KEYS = (
    "level/rough_amp_m",
    "level/slope_deg",
    "level/step_height_m",
    "level/gap_width_m",
    "level/block_height_m",
    "level/genome_dist",
)
GRID_POINTS = 200
HISTORY_WORKERS = 16
# composition is logged every iteration and PLR alternates replay and new
# levels, so it is smoothed with a running mean over this many grid points
SMOOTH = 9
# tab10 has ten colours for fourteen methods; the PAIRED family is dashed
_TAB10 = plt.get_cmap("tab10")
_EXTRA = ("black", "goldenrod", "darkviolet", "darkslategray")
COLORS = dict(zip(METHODS, [*map(_TAB10, range(10)), *_EXTRA], strict=True))


def _scan(run: Any, keys: tuple[str, ...]) -> dict[str, list[float]]:
    rows = run.scan_history(keys=["env_steps", *keys], page_size=10_000)
    out: dict[str, list[float]] = {k: [] for k in ("env_steps", *keys)}
    for row in rows:
        # a NaN logs as None; drop the row so every series stays aligned
        if any(row[k] is None for k in out):
            continue
        for k, values in out.items():
            values.append(float(row[k]))
    return out


def _run_history(run: Any) -> dict[str, Any]:
    out = {"eval": _scan(run, EVAL_KEYS), "level": _scan(run, LEVEL_KEYS)}
    print(f"{run.name}: {len(out['eval']['env_steps'])} evals", flush=True)
    return out


def dump_history(prefixes: dict[str, str], path: Path) -> None:
    api = wandb.Api(timeout=60)
    runs = []
    for prefix in prefixes.values():
        name = rf"^{re.escape(prefix)}_.+_seed[0-9]+$"
        runs += api.runs(
            f"{ENTITY}/{PROJECT}",
            filters={"display_name": {"$regex": name}, "state": "finished"},
        )
    # one scan_history per run is network-bound; 300 runs take ~2 h serially
    with ThreadPoolExecutor(max_workers=HISTORY_WORKERS) as pool:
        histories = pool.map(_run_history, runs)
        history = {run.name: h for run, h in zip(runs, histories, strict=True)}
    path.write_text(json.dumps(history))
    print(f"wrote {len(history)} run histories to {path}")


def _runs_of(
    history: dict[str, Any], prefix: str, method: str
) -> list[dict[str, Any]]:
    pattern = re.compile(rf"^{re.escape(prefix)}_{method}_seed[0-9]+$")
    return [h for name, h in sorted(history.items()) if pattern.match(name)]


def _on_grid(
    runs: list[dict[str, Any]], part: str, key: str
) -> tuple[np.ndarray, np.ndarray]:
    end = min(max(r[part]["env_steps"]) for r in runs)
    grid = np.linspace(0.0, end, GRID_POINTS)
    curves = [np.interp(grid, r[part]["env_steps"], r[part][key]) for r in runs]
    return grid, np.stack(curves)


def _smooth(ys: np.ndarray) -> np.ndarray:
    kernel = np.ones(SMOOTH) / SMOOTH
    padded = np.pad(ys, ((0, 0), (SMOOTH // 2, SMOOTH // 2)), mode="edge")
    return np.stack([np.convolve(y, kernel, mode="valid") for y in padded])


def _band(ax: Any, grid: np.ndarray, ys: np.ndarray, method: str) -> None:
    color = COLORS[method]
    style = "--" if method.startswith("paired") else "-"
    ax.plot(
        grid, ys.mean(axis=0), style, color=color, lw=1.2, label=METHODS[method]
    )
    ax.fill_between(
        grid, ys.min(axis=0), ys.max(axis=0), color=color, alpha=0.15
    )


def plot_curves(
    history: dict[str, Any], prefixes: dict[str, str], out: Path
) -> None:
    robots = list(prefixes)
    fig, axes = plt.subplots(
        len(EVAL_KEYS),
        len(robots),
        figsize=(1.6 * len(robots), 2.0 * len(EVAL_KEYS) + 0.6),
        squeeze=False,
        sharey="row",
    )
    for i, robot in enumerate(robots):
        for j, key in enumerate(EVAL_KEYS):
            ax = axes[j, i]
            for method in METHODS:
                runs = _runs_of(history, prefixes[robot], method)
                if runs:
                    grid, ys = _on_grid(runs, "eval", key)
                    _band(ax, grid / 1e6, 100 * ys, method)
            d = key.split("/")[1][1:]
            ax.set_title(f"{ROBOT_LABEL[robot]}, $d={d}$", fontsize=9)
            ax.set_xlabel("env steps (M)", fontsize=8)
            ax.set_ylabel("success (%)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "curves.pdf")
    plt.close(fig)


def _unit(name: str) -> tuple[str, float]:
    if name.endswith("_m"):
        return name.removesuffix("_m").replace("_", " ") + " (cm)", 100.0
    if name.endswith("_deg"):
        return name.removesuffix("_deg").replace("_", " ") + " (deg)", 1.0
    return name.replace("_", " "), 1.0


def plot_composition(
    history: dict[str, Any], prefixes: dict[str, str], out: Path
) -> None:
    robots = list(prefixes)
    fig, axes = plt.subplots(
        len(robots),
        len(LEVEL_KEYS),
        figsize=(7.5, 1.6 * len(robots) + 0.5),
        squeeze=False,
    )
    for i, robot in enumerate(robots):
        for j, key in enumerate(LEVEL_KEYS):
            ax = axes[i, j]
            name = key.removeprefix("level/")
            label, factor = _unit(name)
            # Go1 logs physical lengths at its own size; report in Wojtek units
            if name.endswith("_m"):
                factor /= ROBOT_SCALE[robot]
            for method in METHODS:
                runs = _runs_of(history, prefixes[robot], method)
                if runs:
                    grid, ys = _on_grid(runs, "level", key)
                    _band(ax, grid / 1e6, factor * _smooth(ys), method)
            ax.set_title(label, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ROBOT_LABEL[robot], fontsize=9)
            if i == len(robots) - 1:
                ax.set_xlabel("env steps (M)", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.3)
    fig.savefig(out / "composition.pdf")
    plt.close(fig)


def figures(cfg: PaperFiguresConfig) -> None:
    if cfg.dump_history:
        dump_history(run_prefixes(cfg.prefixes), Path(cfg.dump_history))
        return
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.history:
        prefixes = run_prefixes(cfg.prefixes)
        history = json.loads(Path(cfg.history).read_text())
        plot_curves(history, prefixes, out)
        plot_composition(history, prefixes, out)
    if cfg.terrains:
        # pulls in jax and the env stack, which the curves do not need
        from paper.terrains import plot_terrains

        plot_terrains(out, cfg.arena)
