# CLAUDE.md

Project guide for Claude. The global `~/.claude/CLAUDE.md` conventions apply;
this file only records where things live and where this repo deviates. **The
docs are the source of truth — read the relevant one before working, and update
it when the code changes.**

## Read this doc before touching this area

| Task / area | Read first |
|---|---|
| Anything — get oriented (methods, robots, metrics, pipeline) | [docs/overview.md](docs/overview.md) |
| Reproducing the paper, run names, cluster values | [README.md](README.md#reproduce-the-paper) |
| Install, deps/pins, building assets, running | [docs/setup.md](docs/setup.md) |
| The env, tasks, terrain, `FunctionalEnv`, the shared tiled arena | [docs/environment.md](docs/environment.md) |
| `training/` (trainer, curriculum modes), heightfield levels, `evaluation/`, `rendering/`, `paper/` | [docs/ued-accel.md](docs/ued-accel.md) |
| Playground robots (`robots/`: Go1, Barkour vB, Spot, Go2), their scale and scene | [docs/ued-accel.md](docs/ued-accel.md#robots) |
| Hydra configs (`conf/`, `config/schema.py`), any CLI argument | [docs/configuration.md](docs/configuration.md) |
| Layout rules, tooling, adding features, gotchas | [docs/development.md](docs/development.md) |
| Train-from-template, ONNX export, the generated deployment README (`export/`, `conf/policy/`) | [docs/export.md](docs/export.md) |
| Running training/eval on the cluster | `slurm/*.sbatch`, [README.md](README.md#slurm-job-interface) |

## Where code lives

- `src/wojtek_env/` — extracted env library. **Do not add files here**; it is
  ruff-excluded and hatchling-packaged. The one edit point is
  `envs/registry.py` (register a task).
- Research code, one package per role (`docs/overview.md` repo map):
  - `src/scripts/` — thin Hydra entry points only (parse config, build, run,
    log); the `just` recipes and `slurm/` call them by path.
  - `src/training/` — the trainer behind `scripts/brax_ued.py` (brax PPO
    student, unroll, lane iteration, curricula, PAIRED, buffer, train loop).
  - `src/evaluation/` — `eval_brax` / `eval_sim2sim` logic (suite, stats,
    rollouts, CPU MuJoCo physics and its variants) and the level-by-level
    domination comparison of eval dumps (`compare_domination`).
  - `src/env/` — `FunctionalEnv` and the env factory/overrides;
    `src/hfield/` — levels and the tiled arena; `src/config/` — schemas.
  - `src/export/` — checkpoint + config → ONNX policy and its README:
    `contract` (built from the env), `onnx_graph`, `verify`, `readme`, `run`.
  - `src/rendering/`, `src/paper/`, `src/diagnostics/` — videos, paper
    artefacts, arena check (`check_hfield_batch.py`) and throughput
    profile (`just profile`).
- `src/robots/` — Playground quadrupeds on the tiled arena: `terrain.py`
  (`TerrainRobot`, the shared scene/spawn/fall/scan/commands) plus one file
  per robot with its constants and differences; `registry.py` lists them,
  `scale.py` holds the robot-size scaling.
  A new robot = a file, a `conf/robot/<name>.yaml`, a `Robot` enum value, a
  `TERRAIN_ROBOTS` entry, `ROBOT_LABEL` (`paper/labels.py`) and
  `prefixes` entries (default `ued_<robot>`).
- `conf/` — Hydra YAML, one primary per entry point plus the `robot`, `mode`,
  `level_space`, `arena`, `network` groups and the optional `policy` group
  (whole-policy definitions, `conf/policy/template.yaml`);
  `src/config/schema.py` types each primary.
  A new curriculum = a `conf/mode/<name>.yaml` and a `METHODS` label
  (`paper/labels.py`).
- `slurm/` — job arrays (train, final eval, sim2sim, asset build); cluster
  values (`#SBATCH` lines, `REPO`, `PDW`, `WB_*`) at the top of each.

## Project-specific rules (override / complement the global)

- **Python 3.11–3.12**, not 3.13 — forced by the `jax==0.9.2` /
  `mujoco-mjx==3.10.0` / `warp-lang==1.13.0` pins. Do not bump those without
  reading [docs/setup.md](docs/setup.md#pinned-dependencies-do-not-bump-blindly).
- **Namespace packages** — no `__init__.py` in research code; imports are
  absolute via `PYTHONPATH=src`.
- **Generated assets are gitignored** — run `just build` and
  `just build-accel-arena` before anything that loads the robot / `accel` arena.
- `just lint` is ruff-only; run `ty check` manually.
- **Hydra, no argparse** — every research entry point (`src/scripts/`,
  `hfield.arena`) is `@hydra.main` over `conf/<name>.yaml` (the frozen
  `wojtek_env` tools keep their own flags). YAML
  holds the values, the schema dataclass (fields `= MISSING`) types them:
  a new option is a schema field plus a YAML value, never a default in
  code. Wojtek runs and `just build-accel-arena` must read the same `arena`
  group (`check_arena` enforces it).
- **Shared tiled arena** — every level of an iteration sits on its own tile
  of one `hfield_data`; never vmap `hfield_data` per env (warp collides every
  env with the first one's). Warp needs `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (the justfile and `slurm/` export it). The Playground robots run on jax
  only: their foot contacts diverge on warp far from the arena origin.

- **Paper state** — docs and comments describe the code as it is: no
  version names, job IDs or history; results live in wandb and the paper.
- **Run names** — `ued_<robot>_<mode>_seed<N>` (`slurm/`), evals
  `<run>_final_eval`; `paper_*` and `domination` parse them via
  `prefixes.<robot>`.

## After changing code

Per the global "update docs after major changes" rule: update the matching
`docs/<topic>.md` (and this table if a topic moves), plus the `README.md` recipe
table if the `justfile` changed. Keep docs as bullet points, factual, no padding.
