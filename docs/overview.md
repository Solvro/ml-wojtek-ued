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
| **Research code** | `src/{config,env,robots,hfield,training,evaluation,rendering,paper,diagnostics}/`, `src/scripts/` | brax-PPO trainer, ACCEL/PLR curriculum, heightfield levels, eval + paper tooling | ruff + ty enforced |

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
    hfield_config.py HFieldConfig (level genome envelope, the `arena` group)
    schema.py        schema dataclass per entry point, enums, CONF_DIR
  env/               wojtek_env / Playground robots as jit/vmap training envs
    functional.py    FunctionalEnv: pure reset/step on the shared arena; Obs
    factory.py       make_env (Wojtek or a Playground robot), check_wojtek_arena
    overrides.py     shared env override dicts (movement / terrain / arena)
    travel.py        TravelBounds: how far an env can read terrain per level
    divergence.py    end_diverged: end episodes whose physics blew up
    height_lookup.py Wojtek's terrain lookup rebuilt from hfield_data
    warp.py          warp backend memory checks and graph cache
  robots/
    terrain.py       TerrainRobot: arena scene, spawn, fall, scan, commands
    go1.py           Playground Go1 on the tiled arena
    barkour.py       Barkour vB
    spot.py          Spot
    go2.py           Go2 (Go1's env on the menagerie model, feet-only rewrite)
    registry.py      TERRAIN_ROBOTS: robot name -> class
    scale.py         ROBOT_SCALE, hfield_config, min_travel (robot size)
  hfield/            heightfield levels  (see ued-accel.md)
    manager.py       HFieldManager: generate / mutate / upsample a level
    level_space.py   level spaces (uniform grid, 9-parameter `param`)
    arena.py         write the static tiled "accel" MJCF arena assets
    tiles.py         TileLayout: one level per tile of the shared arena
  training/          the trainer behind scripts/brax_ued.py
    ppo.py           brax PPO hyperparameters, networks
    student.py       Student: params, optimizer, jitted update / regret / BC KL
    unroll.py        jitted policy rollout on the shared arena
    context.py       TrainContext (everything build makes), KeyStream
    build.py         build a TrainContext from TrainConfig, check_config
    iteration.py     run_levels: one outer iteration over the lanes
    curricula.py     DR / ACCEL / PLR / PAIRED / PAIRED+Evo iterations
    buffer.py        LevelBuffer (rank + staleness replay)
    adversary.py     PAIRED adversary (single-step PPO on regret)
    generator.py     LevelGenerator network (PAIRED adversary policy)
    paired.py        PAIRED regret, student BC KL, BC weights
    evaluate.py      in-training eval on uniform levels
    metrics.py       level composition of the levels trained on
    train.py         the train loop: wandb, logging, checkpoint
  evaluation/        checkpoint evaluation behind eval_brax / eval_sim2sim
    rollout.py       EvalContext, chunked deterministic rollouts
    suite.py         held-out structured terrain suite
    stats.py         cross-fitted CVaR, bootstrap CIs, capability envelopes
    final_eval.py    eval_brax's level sets and metrics
    physics_variants.py  sim2sim's CPU MuJoCo model transforms
    cpu_physics.py   CpuPhysics: CPU MuJoCo stepping under the jax env
    sim2sim.py       sim2sim's level sets, metrics, build
  rendering/
    scene.py         one-tile render model, tracking camera
    video.py         rollout + mp4 clips (render_video)
  paper/
    runs.py          eval/run loading, solvable subset, wandb summaries
    labels.py        robot / method labels
    latex.py         booktabs cells and tabulars
    tables.py        main / envelopes / compute tables, CSV
    figures.py       curves and composition figures
    terrains.py      rendered example levels
  diagnostics/
    arena_check.py   per-backend shared-arena sanity check
    throughput.py    throughput profile of a training iteration
  scripts/           thin Hydra entry points (`just` recipes, slurm/)
    brax_ued.py          train                       (`just train`)
    eval_brax.py         final eval of a checkpoint  (`just eval`)
    eval_sim2sim.py      CPU MuJoCo variants         (`just sim2sim`)
    render_video.py      mp4 clips of a checkpoint   (`just render-video`)
    paper_tables.py / paper_figures.py               (`just paper-*`)
    profile_brax_ued.py  throughput profile          (`just profile`)
    check_hfield_batch.py  shared-arena check
main.py              random-agent MuJoCo-viewer preview
conf/                Hydra YAML: one primary per entry point + groups
                     robot/ mode/ level_space/ arena/  (see configuration.md)
```

## End-to-end data flow

**Train** (`scripts/brax_ued.py` → `training/`): `FunctionalEnv` wraps a task env on the
shared tiled arena → each iteration writes `num_lanes` levels into tiles of one
`hfield_data` (`TileLayout.assemble`), resets `envs_per_lane` envs per tile with
a fresh physics DR draw, and steps them in 20-step unrolls with brax's PPO
(networks, running obs normalizer, `compute_ppo_loss`) → per-lane regret
(mean positive GAE advantage) → `mode=` (`conf/mode/`):
- `dr` / `dr_easy` / `dr_mixed` — fresh uniform levels every iteration (full difficulty / easy / mixed).
- `accel` — replay high-regret buffer levels (training), mutate them, and score
  mutations + fresh easy levels on explore iterations (no training).
- `plr` — ACCEL's buffer and replay without mutation.
- `paired` (`paired_uni` / `paired_bi`: + student BC) — a learned
  `LevelGenerator` proposes levels; two students, PAIRED regret.
- `paired_evo` — ACCEL's buffer and mutation scored by PAIRED regret.

Levels are `coarse_n × coarse_n` control-point grids (`grid` space) or 9 terrain
parameters (`param` space), upsampled to the MJX heightfield. Full walkthrough
in [ued-accel.md](ued-accel.md).

**Eval** (`evaluation/`): `eval_brax.py` scores a saved `<run>.pkl` on fixed terrain suites and
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
- `conf/` groups and every config field → [configuration.md](configuration.md)
- Tooling, imports, conventions → [development.md](development.md)
