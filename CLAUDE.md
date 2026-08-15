# CLAUDE.md

Project guide for Claude. The global `~/.claude/CLAUDE.md` conventions apply;
this file only records where things live and where this repo deviates. **The
docs are the source of truth — read the relevant one before working, and update
it when the code changes.**

## Read this doc before touching this area

| Task / area | Read first |
|---|---|
| Anything — get oriented | [docs/overview.md](docs/overview.md) |
| Install, deps/pins, building assets, running | [docs/setup.md](docs/setup.md) |
| The env, tasks, terrain, `FunctionalEnv`, hfield swapping | [docs/environment.md](docs/environment.md) |
| PPO model / rollout / loss / checkpoints / logging | [docs/ppo.md](docs/ppo.md) |
| ACCEL curriculum, heightfield levels, convergence | [docs/ued-accel.md](docs/ued-accel.md) |
| Config fields, Hydra overrides | [docs/configuration.md](docs/configuration.md) |
| Layout rules, tooling, adding features, gotchas | [docs/development.md](docs/development.md) |

## Where code lives

- `src/wojtek_env/` — extracted env library. **Do not add files here**; it is
  ruff-excluded and hatchling-packaged. The one edit point is
  `envs/registry.py` (register a task).
- `src/{config,modules,utils,hfield,ued}/`, `src/scripts/` — research code.
  Add new work here as sibling packages.
- `src/ued/` — the ACCEL algorithm, split by role (`types`, `regret`, `levels`,
  `stopper`, `curriculum`). Keep the split; don't recreate a `helpers` grab-bag.

## Project-specific rules (override / complement the global)

- **Python 3.11–3.12**, not 3.13 — forced by the `jax==0.9.2` /
  `mujoco-mjx==3.10.0` / `warp-lang==1.13.0` pins. Do not bump those without
  reading [docs/setup.md](docs/setup.md#pinned-dependencies-do-not-bump-blindly).
- **Namespace packages** — no `__init__.py` in research code; imports are
  absolute via `PYTHONPATH=src`.
- **Generated assets are gitignored** — run `just build` and
  `just build-accel-arena` before anything that loads the robot / `accel` arena.
- `just lint` is ruff-only; run `ty check` manually.

## After changing code

Per the global "update docs after major changes" rule: update the matching
`docs/<topic>.md` (and this table if a topic moves), plus the `README.md` recipe
table if the `justfile` changed. Keep docs as bullet points, factual, no padding.
