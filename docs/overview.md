# Overview

Research codebase for **Unsupervised Environment Design (UED)** — PAIRED/ACCEL —
on the Wojtek quadruped. A student policy is trained with PPO while an
adversarial curriculum invents and replays heightfield "levels" it finds hard.

## Two layers

The repo is split into a frozen env library and the research code that consumes
it. They never mix — research code lives in sibling packages under `src/`, not
inside `wojtek_env`.

| Layer | Location | Role | Tooling |
|---|---|---|---|
| **Env library** | `src/wojtek_env/` | MJX sim env, procedural terrain, Gymnasium wrapper, build/check CLIs | packaged (hatchling), **excluded from ruff** |
| **Research code** | `src/{config,modules,utils,hfield,robots}/`, `src/scripts/` | brax-PPO trainer, ACCEL/PLR curriculum, heightfield levels, eval + paper tooling | ruff + ty enforced |

`src/wojtek_env/` is extracted from an upstream robot stack (w01-tek) and treated
as a stable dependency: env only, no trainer, no ROS, no deployment. Details in
[environment.md](environment.md).

## Repo map

```
src/
  wojtek_env/        env library (see environment.md)
    envs/            MJX envs: joystick, getup, jump  (registry.make_env)
    terrain/         procedural terrain generator + curriculum wrapper
    gymnasium/       WojtekGymEnv
    tools/           build_model / build_terrain / check_* CLIs
    paths.py         repo-relative asset paths, robot constants
  config/            Hydra structured configs  (see configuration.md)
    hfield_config.py HFieldConfig (level genome envelope)
    arena_config.py  ArenaConfig for the arena builder
  modules/
    generator.py     LevelGenerator (PAIRED adversary)
  robots/
    go1.py           Playground Go1 on the tiled arena
  utils/
    env.py           FunctionalEnv: bridges wojtek_env into jit/vmap training; Obs
    env_overrides.py shared env override dicts (movement / terrain / arena)
  hfield/            heightfield levels  (see ued-accel.md)
    manager.py       HFieldManager: generate / mutate / upsample a level
    level_space.py   level spaces (uniform grid, 9-parameter `param`)
    arena.py         write the static tiled "accel" MJCF arena assets
    tiles.py         TileLayout: one level per tile of the shared arena
  scripts/
    brax_ued.py          the trainer: brax PPO student, DR / ACCEL / PLR / PAIRED
    eval_brax.py         final eval of a checkpoint on suites + uniform levels
    eval_sim2sim.py      the same checkpoint on CPU MuJoCo variants
    render_video.py      mp4 clips of a checkpoint     (`just render-video`)
    paper_tables.py / paper_figures.py / paper_terrains.py   paper artefacts
    profile_brax_ued.py  throughput profile of brax_ued
    check_hfield_batch.py  per-backend shared-arena sanity check
main.py              random-agent MuJoCo-viewer preview
```

## End-to-end data flow

**Train** (`scripts/brax_ued.py`): `FunctionalEnv` wraps a task env on the
shared tiled arena → each iteration writes `num_lanes` levels into tiles of one
`hfield_data` (`TileLayout.assemble`), resets `envs_per_lane` envs per tile with
a fresh physics DR draw, and steps them in 20-step unrolls with brax's PPO
(networks, running obs normalizer, `compute_ppo_loss`) → per-lane regret
(mean positive GAE advantage) → `--mode`:
- `dr` / `dr_easy` / `dr_mixed` — fresh uniform levels every iteration (full difficulty / easy / mixed).
- `accel` — replay high-regret buffer levels (training), mutate them, and score
  mutations + fresh easy levels on explore iterations (no training).
- `plr` — ACCEL's buffer and replay without mutation.
- `paired` — a learned `LevelGenerator` proposes levels.

Levels are `coarse_n × coarse_n` control-point grids (`grid` space) or 9 terrain
parameters (`param` space), upsampled to the MJX heightfield. Full walkthrough
in [ued-accel.md](ued-accel.md).

**Eval**: `eval_brax.py` scores a saved `<run>.pkl` on fixed terrain suites and
uniform levels; `eval_sim2sim.py` replays it on CPU MuJoCo under model/solver
variants; `render_video.py` renders clips. Details in [ued-accel.md](ued-accel.md).

## Glossary

- **Level / genome** — a `coarse_n × coarse_n` array (or 9 terrain
  parameters, `param` space) in `[0,1]`; rendered to a heightfield patch and
  written into its tile of the shared arena's `hfield_data`, the terrain the
  robot walks on.
- **Regret** — positive value loss (Eq. 5): how much the critic under-estimated
  return, i.e. how "surprisingly hard" a level was. The curriculum's score.
- **Student** — the brax PPO policy being trained.

## Where to read next

- Get it running → [setup.md](setup.md)
- The env, tasks, terrain, hfield-swap trick → [environment.md](environment.md)
- The curriculum, heightfield levels, trainer + eval scripts → [ued-accel.md](ued-accel.md)
- Arena config fields → [configuration.md](configuration.md)
- Tooling, imports, conventions → [development.md](development.md)
