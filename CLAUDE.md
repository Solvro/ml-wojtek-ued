# CLAUDE.md

Quadruped UED Benchmark: Unsupervised Environment Design for quadruped
locomotion on procedural terrain (12 curricula x 5 robots). The global
`~/.claude/CLAUDE.md` applies; this file records where things live and where
this repo deviates. **Read the matching doc before touching an area; update it
when the code changes.** Page map: [docs/index.md](docs/index.md).

## Docs

| Task / area | Read first |
|---|---|
| Get oriented: methods, robots, metrics, pipeline, repo map | [docs/overview.md](docs/overview.md) |
| Reproduce the paper, quickstart | [README.md](README.md) |
| Install, pins, assets | [docs/setup.md](docs/setup.md) |
| `wojtek_env`, `FunctionalEnv`, the tiled arena, backends | [docs/environment.md](docs/environment.md) |
| `hfield/`: levels, level spaces, arena assets | [docs/levels.md](docs/levels.md) |
| `training/`: trainer, curricula, PAIRED, throughput | [docs/training.md](docs/training.md) |
| `robots/`: Go1, Go2, Barkour vB, Spot | [docs/robots.md](docs/robots.md) |
| `evaluation/`, `rendering/`: final eval, sim2sim, videos | [docs/evaluation.md](docs/evaluation.md) |
| `paper/`: tables, figures, domination | [docs/results.md](docs/results.md) |
| `export/`, `conf/policy/`: train-from-template, ONNX export, the deployment README | [docs/export.md](docs/export.md) |
| `conf/`, `config/schema.py`, any CLI argument | [docs/configuration.md](docs/configuration.md) |
| `slurm/`: job variables, per-robot budgets, cluster values | [docs/slurm.md](docs/slurm.md) |
| Layout rules, recipes, tests, adding features, gotchas | [docs/development.md](docs/development.md) |

## Where code lives

- `src/wojtek_env/` — extracted Wojtek env library (frozen).
- `src/scripts/` — thin Hydra entry points; `just` and `slurm/` call them.
- `src/config/` — structured-config schemas (`schema.py`), `HFieldConfig`.
- `src/env/` — `FunctionalEnv`, env factory/overrides, warp helpers.
- `src/robots/` — Playground quadrupeds on the tiled arena.
- `src/hfield/` — levels, level spaces, tiled arena.
- `src/training/` — brax-PPO student, curricula, PAIRED, train loop.
- `src/export/` — checkpoint + config → ONNX policy and its README; the contract is built from the env (Playground obs layout measured by shifting input signals, `playground_obs.py`), export env always on jax.
- `src/evaluation/` — final eval, sim2sim, domination.
- `src/rendering/` — mp4 clips.
- `src/paper/` — tables, statistics (rliable), figures, labels.
- `src/diagnostics/` — arena check, throughput profile.
- `conf/` — Hydra YAML (groups `robot`, `mode`, `level_space`, `arena`,
  `network`, optional `policy`); `slurm/` — job arrays; `tests/` mirrors `src/`.

## Hard rules

- **Python 3.11–3.12** (`.python-version` 3.12), not 3.13: forced by the
  `jax==0.9.2` / `mujoco-mjx==3.10.0` / `warp-lang==1.13.0` pins. Do not bump
  them without [docs/setup.md](docs/setup.md#pinned-dependencies-do-not-bump-blindly).
- **`src/wojtek_env/` is frozen** — add no files; ruff-excluded. The one edit
  point is `envs/registry.py` (register a task).
- **Namespace packages** — no `__init__.py` in research code; absolute imports
  via `PYTHONPATH=src` (exported by the `justfile`).
- **Hydra only, no argparse** — every research entry point is `@hydra.main`
  over `conf/<name>.yaml`. A new option = a schema field (`= MISSING`) plus a
  YAML value, never a default in code. Exception: `RobotConfig`'s
  `kp` / `kd` / `damping` / `scale` default to `None` (= the robot's own).
- **Tiled arena** — every level of an iteration sits on its own tile of one
  shared `hfield_data`; never vmap `hfield_data` (warp collides every env with
  the first one's). Warp needs `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The
  Playground robots run on jax only. Wojtek runs and
  `just build-accel-arena` must read the same `arena` group.
- **Generated assets are gitignored** — `just build` and
  `just build-accel-arena` before anything that loads the robot or arena.
- **Run names** — `ued_<robot>_<mode>_seed<N>`, evals `<run>_final_eval`;
  `paper_*` and `domination` parse them via `prefixes.<robot>`.
- **Docs describe the code as it is** — no version names, job IDs or history;
  results live in wandb and the paper. Heading anchors are GitHub slugs (no
  `{#id}`).
- `just lint` is ruff-only; run `ty check` manually.

## After changing code

Update the matching `docs/<topic>.md`, this table if a topic moves, and
`docs/index.md`; the README only for commands of the paper pipeline. Bullet
points, factual, no padding.
