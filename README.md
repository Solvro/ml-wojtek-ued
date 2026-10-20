# Wojtek UED Research

Unsupervised Environment Design for quadruped locomotion: a brax-PPO student
trained against 12 curricula (DR, PLR⊥, SFL, ACCEL, PAIRED and variants) of
procedurally generated heightfield levels, on the W01-TEK quadruped and four
MuJoCo Playground robots. Built on `wojtek_env`, an extracted MJX env +
procedural terrain generator + Gymnasium interface.

## Documentation

| Doc | Read it for |
|---|---|
| [overview.md](docs/overview.md) | methods, robots, level space, metrics, pipeline, repo map |
| [setup.md](docs/setup.md) | install, dependency pins, building assets, outputs |
| [environment.md](docs/environment.md) | `wojtek_env` tasks/terrain/gym, `FunctionalEnv`, the shared tiled arena |
| [ued-accel.md](docs/ued-accel.md) | levels, the trainer + curricula, eval, sim2sim, render, paper scripts |
| [configuration.md](docs/configuration.md) | Hydra configs in `conf/`: every entry point, group and field |
| [development.md](docs/development.md) | layout rules, tooling, conventions, gotchas |

## Reproduce the paper

The reported numbers: 12 methods (`conf/mode/*`) x 5 robots
(`conf/robot/*`) x seeds 0-4 = **300 training runs**, each followed by one
final eval, then the domination comparison, tables and figures.

| setting | value |
|---|---|
| training | 300M env steps (`num_timesteps`), 256 levels x 32 envs per iteration, 300-step levels, `level_space=param`, `arena=benchmark`, `backend=auto` (warp on a GPU) |
| final eval | `conf/eval.yaml` defaults: 2048 levels per random set, 4 episodes x 2 passes, 64 levels per suite cell and flat, 300 steps, success = no fall and >= 1 m x robot scale, CVaR 10 / 25 %, seed 1234, same backend as training |
| run names | `ued_<robot>_<mode>_seed<N>`, evals `ued_<robot>_<mode>_seed<N>_final_eval` |

### 1. Environment

- Linux x86_64, one NVIDIA GPU per run with a CUDA 12 driver (H100 used);
  macOS installs CPU jax only (tests, CPU smoke runs, figures).
- Python 3.12, pinned in `.python-version` (`requires-python >=3.11,<3.13`), [uv](https://docs.astral.sh/uv/),
  [just](https://just.systems/), `ffmpeg` for videos.
- Exact versions from `uv.lock` (jax/jaxlib 0.9.2 + `jax[cuda12]`,
  mujoco 3.11.0, mujoco-mjx 3.10.0, warp-lang 1.13.0, brax 0.14.2,
  playground 0.2.0):

  ```bash
  uv sync --locked --extra dev
  ```

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is required by the warp backend; the
  `justfile` and `slurm/*.sbatch` export it.
- A wandb account: training, eval and the paper dumps log to / read from
  `entity` / `project`.

### 2. Assets

Generated and gitignored; build once per checkout (or frozen copy):

```bash
just build               # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
just build-accel-arena   # tiled "accel" arena, conf/arena.yaml (arena=benchmark)
# Playground robots (go1, barkour, spot, go2): mujoco_menagerie at
# Playground's pinned commit, cloned into the venv
uv run python -c "from mujoco_playground._src.mjx_env import \
ensure_menagerie_exists; ensure_menagerie_exists()"
```

### 3. Training grid (SLURM)

On the cluster, in a clone at the paper commit with the venv synced
(`REPO` in the sbatch files). Jobs run from a frozen copy so later edits do
not reach queued jobs:

```bash
PDW=<your PD dir>                     # as in slurm/*.sbatch
COPY="$PDW/copies/$(git rev-parse --short HEAD)"
mkdir -p "$PDW/logs"                  # #SBATCH --output dir; SLURM does not create it
mkdir -p "$COPY" && git archive HEAD src conf assets | tar -x -C "$COPY"
BUILD=$(SRC="$COPY/src" sbatch --parsable slurm/build_assets.sbatch)

MODES="dr dr_easy dr_mixed plr sfl accel accel_grow accel_sfl paired paired_uni paired_bi paired_evo"
for ROBOT in wojtek go1 barkour spot go2; do
  SRC="$COPY/src" MODES="$MODES" NSEEDS=5 ROBOT="$ROBOT" TAGS=paper \
    sbatch --dependency="afterok:$BUILD" --array=0-59 slurm/brax_ued.sbatch
done
```

- One array per robot: task = mode index x 5 + seed, 60 tasks.
- Checkpoints: `$PDW/artifacts/brax_ued/ued_<robot>_<mode>_seed<N>.pkl`.
- Cost, from Wojtek's measured warp throughput on an H100
  ([ued-accel.md](docs/ued-accel.md#throughput)): ~0.5 GPU-h per run, ~1 GPU-h
  for the 4 PAIRED variants (two students) → ~8 GPU-h per robot x seed,
  ~200 GPU-h for the grid. Playground robots are not profiled.

### 4. Final evals

```bash
for ROBOT in wojtek go1 barkour spot go2; do
  SRC="$COPY/src" MODES="$MODES" NSEEDS=5 ROBOT="$ROBOT" TAGS=paper \
    sbatch --array=0-59 slurm/eval_brax.sbatch
done
```

- Writes `$PDW/runs/final_eval/ued_<robot>_<mode>_seed<N>_final_eval.{json,npz}`
  and the metrics to a wandb run of that name; ~8.5 min per Wojtek eval on
  warp (~45 GPU-h for 300).
- Optional transfer test: `slurm/eval_sim2sim.sbatch` (same variables, CPU
  partition), [ued-accel.md](docs/ued-accel.md#sim2sim).

Copy the eval dir next to the repo (e.g. `rsync` it to
`artifacts/final_eval`); the rest runs on a laptop.

### 5. Domination comparison

```bash
just domination eval_dir=artifacts/final_eval                  # ACCEL vs DR
just domination eval_dir=artifacts/final_eval method_a=plr method_b=dr \
  out=artifacts/paper/domination_plr_dr
```

- In: `*_final_eval.npz` (`param_uniform` levels); out:
  `artifacts/paper/domination/` (`domination.json`,
  `domination_<a>_vs_<b>[_<robot>].{pdf,png}`, `domination_matrix.pdf`).

### 6. Paper tables and figures

```bash
just paper-tables dump_wandb=artifacts/paper/summaries.json
just paper-tables eval_dir=artifacts/final_eval \
  train_summaries=artifacts/paper/summaries.json
just paper-figures dump_history=artifacts/paper/history.json
just paper-figures history=artifacts/paper/history.json
JAX_PLATFORMS=cpu just paper-figures terrains=true
```

| output | from |
|---|---|
| `artifacts/paper/tables/main.tex`, `envelopes.tex`, `results.csv` | final evals |
| `artifacts/paper/tables/compute.tex` | wandb run summaries (`summaries.json`) |
| `artifacts/paper/figs/curves.pdf`, `composition.pdf` | wandb histories (`history.json`) |
| `artifacts/paper/figs/terrains.pdf`, `robots.pdf` | rendered levels (built assets) |

- `dump_wandb` / `dump_history` read the wandb project in `src/paper/runs.py`
  (`ENTITY`, `PROJECT`).
- A robot without runs: `prefixes.<robot>=null`.
- Tables are bare booktabs `tabular`s for the paper (Overleaf), figures go
  into its `figs/`.

### A single run locally

```bash
just train mode=accel robot=wojtek seed=0 name=ued_wojtek_accel_seed0 \
  checkpoint_dir=artifacts/checkpoints
just eval checkpoint=artifacts/checkpoints/ued_wojtek_accel_seed0.pkl \
  name=ued_wojtek_accel_seed0_final_eval out_dir=artifacts/final_eval
just train --cfg job mode=accel name=x    # print the composed config
```

CPU smoke run (~4 min on a laptop, no GPU, wandb offline):

```bash
WANDB_MODE=offline JAX_PLATFORMS=cpu just train mode=accel name=smoke \
  backend=jax num_lanes=2 envs_per_lane=16 level_steps=20 eval_levels=2 \
  eval_every=1 num_timesteps=1500 mode.buffer_size=8 \
  checkpoint_dir=artifacts/checkpoints
```

### SLURM job interface

Env vars of `slurm/brax_ued.sbatch`, `eval_brax.sbatch`,
`eval_sim2sim.sbatch` (turned into Hydra overrides):

| variable | default | meaning |
|---|---|---|
| `MODES` | `dr accel` (train; required for evals) | `conf/mode` options; train also takes `<mode>-s<scale>` = `mode.generator_scale=<scale>` |
| `NSEEDS` / `SEED0` | `3` / `0` | task = mode index x `NSEEDS` + offset, seed = `SEED0` + offset; array size `NSEEDS` x modes |
| `ROBOT` | `wojtek` | `conf/robot` option |
| `SPACE` | `param` | `conf/level_space` option (train, eval_brax) |
| `PREFIX` | `ued` | run name `<PREFIX>_<ROBOT>_<mode>_seed<N>` |
| `BACKEND` | `auto` | `auto` / `jax` / `warp` (train, eval_brax) |
| `NUM_TIMESTEPS` | `300000000` | train |
| `TAGS` | — | extra wandb tags, space-separated |
| `SRC` | `$REPO/src` | `src/` of a frozen copy with built `assets/` and `conf/` beside it |

### Cluster-specific values

Set to WCSS (Lem, H100) in every `slurm/*.sbatch`; another lab changes:

| where | value |
|---|---|
| `#SBATCH --account` | compute grant |
| `#SBATCH --partition`, `--gres` | GPU partition / GPU type (`build_assets`, `eval_sim2sim`: a CPU partition) |
| `#SBATCH --output`, `--error` | log directory |
| `REPO` | clone with the synced `.venv` |
| `PDW` | work dir: checkpoints `artifacts/brax_ued/`, evals `runs/final_eval/`, `runs/sim2sim/` |
| `WB_ENTITY`, `WB_PROJECT` | wandb entity / project (also `ENTITY`, `PROJECT` in `src/paper/runs.py`) |

## Train and export a policy

A deployable model for the real robot, next to the benchmark: define the
run in one YAML, train one policy, export it to ONNX.

```bash
cp conf/policy/template.yaml conf/policy/my_robot.yaml   # robot, mode, terrain, PPO, network
just train-export policy=my_robot                        # train, then export
just export checkpoint=<run>.pkl policy=my_robot         # export an existing checkpoint
```

- Output `artifacts/export/<name>/`: `policy.onnx` (deterministic actor,
  observation normalizer inside, outputs `action` and `joint_targets`),
  `config.yaml` (the full training config), `README.md` (the generated
  contract: observation layout, joints, gains, commands, `onnxruntime`
  example, training metrics).
- Every robot is supported; the example and the README focus on Wojtek.
- Details, cluster `sbatch` example and the verification:
  [docs/export.md](docs/export.md).

## Justfile recipes

`PYTHONPATH=src` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` are exported for
every recipe; `just --list` lists them. `*args` are Hydra overrides of the
recipe's config.

| recipe | does |
|---|---|
| `just sync` | `uv sync --extra dev` |
| `just install` | `uv lock && uv sync --extra dev` (relock, then sync) |
| `just fmt` | `ruff format` + `ruff check --fix` on `src` |
| `just lint` | `ruff check` + `ruff format --check` on `src` |
| `just build` | build the MJX model (`wojtek_mjx.xml` + `scene_mjx.xml`) |
| `just build-accel-arena *args` | write the tiled `accel` arena; `conf/arena.yaml` |
| `just train *args` | train with `brax_ued.py`; `conf/train.yaml` (`mode=` required) |
| `just profile *args` | throughput profile of a training iteration; `conf/profile.yaml` |
| `just eval *args` | final eval of a checkpoint; `conf/eval.yaml` |
| `just sim2sim *args` | the checkpoint on CPU MuJoCo physics variants; `conf/sim2sim.yaml` |
| `just render-video *args` | mp4 clips of a checkpoint on generated levels; `conf/render.yaml` |
| `just paper-tables *args` | LaTeX tables from final evals; `conf/paper_tables.yaml` |
| `just paper-figures *args` | figures from wandb + rendered terrains; `conf/paper_figures.yaml` |
| `just domination *args` | level-by-level domination comparison; `conf/domination.yaml` |

## Env usage

```python
from wojtek_env.gymnasium import WojtekGymEnv

env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
```
