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
| **Research code** | `src/{config,modules,utils,hfield,ued}/`, `src/scripts/` | PPO trainer, ACCEL curriculum, heightfield levels, Hydra config | ruff + ty enforced |

`src/wojtek_env/` is copied from an upstream robot stack (w01-tek, synced at
`e44c1a9`) and treated as a stable dependency: env only, no trainer, no ROS,
no deployment. Details in [environment.md](environment.md#upstream-sync).

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
    ppo_config.py    Config  -> registered as "config"
    accel_config.py  AccelConfig -> registered as "accel_config"
    hfield_config.py HFieldConfig (level genome envelope)
  modules/
    ppo_modules.py   A2C equinox model, Obs / Episode pytrees
  utils/
    env.py           FunctionalEnv: bridges wojtek_env into jit/vmap training
    ppo_helpers.py   rollout, GAE, PPO loss/step, eval  (see ppo.md)
    buffer.py        LevelBuffer (deque of levels)
    checkpoint.py    orbax save/restore + artifacts/<algo>/<run>/ naming + meta.json
    logging.py       wandb or a jsonl fallback Logger
  hfield/            heightfield levels for ACCEL  (see ued-accel.md)
    manager.py       HFieldManager: generate / mutate / upsample a level
    arena.py         write the static tiled "accel" MJCF arena assets
    tiles.py         TileLayout: one level per tile of the shared arena
    playground.py    offscreen render demo of generated levels
  ued/               the ACCEL algorithm  (see ued-accel.md)
    types.py         RolloutFn, Curator, PPOTrainer
    regret.py        positive_value_loss (regret metric)
    levels.py        buffer/level plumbing
    stopper.py       PlateauStopper (convergence)
    curriculum.py    generate/replay branches (Algorithm 1)
  scripts/
    ppo.py           plain PPO on one task            (`just train`)
    ppo_dr.py        PPO + baseline domain randomization (`just train-dr`)
    accel.py         ACCEL curriculum, vanilla         (`just accel`)
    accel_top_k.py   ACCEL curriculum, batched-lanes/top-k (`just accel-top-k`)
    eval.py          evaluate a saved checkpoint       (`just eval --algo <algo>`)
    eval_levels.py   evaluate across generated levels  (`uv run python src/scripts/eval_levels.py --algo <algo>`)
main.py              random-agent MuJoCo-viewer preview
```

## End-to-end data flow

**Plain PPO** (`scripts/ppo.py`): `FunctionalEnv` wraps a task env → `A2C`
rolls out batches (`batch_rollout`) → GAE advantages → clipped PPO epochs
(`run_ppo_epochs`) → checkpoint + log. Fixed `iterations` budget.

**PPO + domain randomization** (`scripts/ppo_dr.py`): identical loop, built on
`FunctionalEnv(domain_randomize=True)`. Each iteration samples one randomized
physics draw per parallel env (`env.sample_dr`, `wojtek_env`'s baseline
friction/mass/gain-kd randomization) and feeds it into `batch_rollout`; eval
still runs on the nominal, non-randomized physics. See
[ppo.md](ppo.md#domain-randomization-scriptsppo_drpy).

**ACCEL** (`scripts/accel.py`): each iteration flips a coin (`decide_replay`):
- **generate** — sample a fresh level from `HFieldManager`, roll out (no policy
  update), score its regret, add to the buffer if hard enough.
- **replay** — sample a stored level, **train the student on it** (PPO), then
  mutate it, score the edited level's regret, curate.

Levels are `coarse_n × coarse_n` control-point grids upsampled to the MJX
heightfield; the buffer keeps the high-regret ones. Convergence is a **regret
plateau**, not a fixed step count. Full walkthrough in [ued-accel.md](ued-accel.md).

**Eval** (`scripts/eval.py`): each trainer checkpoints under
`artifacts/<algo>/<run>/` (`<algo>` is one of `ppo`/`ppo_dr`/`accel`/
`accel_top_k`) and writes a `meta.json` sidecar recording the model shape and
env construction. `just eval --algo <algo>` rebuilds the `A2C` template and
`FunctionalEnv` from that sidecar and restores the latest (or `--run`/`--step`
selected) checkpoint — no Hydra config needed at eval time. Details in
[ppo.md](ppo.md#checkpoints--utilscheckpointpy).

## Glossary

- **Level / genome** — a `coarse_n × coarse_n` array (or 9 terrain
  parameters, `param` space) in `[0,1]`; rendered to a heightfield patch and
  written into its tile of the shared arena's `hfield_data`, the terrain the
  robot walks on.
- **Regret** — positive value loss (Eq. 5): how much the critic under-estimated
  return, i.e. how "surprisingly hard" a level was. The curriculum's score.
- **Curator** — bundles the level buffer + generator + replay/regret thresholds.
- **Student** — the `A2C` policy being trained.

## Where to read next

- Get it running → [setup.md](setup.md)
- The env, tasks, terrain, hfield-swap trick → [environment.md](environment.md)
- PPO model, rollout, loss, checkpoints → [ppo.md](ppo.md)
- The ACCEL curriculum + heightfield levels → [ued-accel.md](ued-accel.md)
- Every config field and Hydra overrides → [configuration.md](configuration.md)
- Tooling, imports, conventions → [development.md](development.md)
