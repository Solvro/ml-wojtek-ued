# Quadruped UED Benchmark

**Unsupervised Environment Design for quadruped locomotion on procedural terrain**

A benchmark of Unsupervised Environment Design (UED) and domain-randomization
curricula for legged locomotion. A brax-PPO student walks on procedurally
generated heightfield levels; the curriculum decides which levels it sees.
12 curricula x 5 quadrupeds, trained, evaluated and compared with one
pipeline. Physics runs on MJX (jax or warp backend).

**Methods** (`mode=`, [details](docs/overview.md#methods)):
DR, DR-easy, DR-mixed, PLR⊥, SFL, ACCEL, ACCEL-grow, ACCEL-SFL, PAIRED,
PAIRED+UniBC, PAIRED+BiBC, PAIRED+Evo.

**Robots** (`robot=`, [details](docs/robots.md)): Wojtek (W01-TEK), Unitree
Go1, Unitree Go2, Google Barkour vB, Boston Dynamics Spot.

## Quickstart

Needs [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).
Python 3.12 is pinned (`.python-version`). macOS and Linux work; training at
scale needs a Linux x86_64 box with an NVIDIA GPU (CUDA 12).

```bash
git clone https://github.com/Solvro/ml-wojtek-ued.git
cd ml-wojtek-ued
just sync                # uv sync --extra dev
```

Build the generated assets (gitignored; once per checkout):

```bash
just build               # Wojtek MJX model
just build-accel-arena   # tiled terrain arena
uv run python -c "from mujoco_playground._src.mjx_env import \
ensure_menagerie_exists; ensure_menagerie_exists()"   # Go1/Go2/Barkour/Spot meshes
```

CPU smoke run (~4 min, no GPU, wandb offline):

```bash
WANDB_MODE=offline JAX_PLATFORMS=cpu just train mode=accel name=smoke \
  backend=jax num_lanes=2 envs_per_lane=16 level_steps=20 eval_levels=2 \
  eval_every=1 num_timesteps=1500 mode.buffer_size=8 \
  checkpoint_dir=artifacts/checkpoints
```

Print any composed config instead of running it:

```bash
just train --cfg job mode=paired robot=go1 name=x
```

`just --list` shows all recipes. Tests: `uv run pytest -q`.

## Reproduce the paper

12 methods (`conf/mode/*`) x 5 robots (`conf/robot/*`) x seeds 0-4 = **300
training runs**, each followed by one final eval, then the domination
comparison, tables and figures.

| robot | env steps (`NUM_TIMESTEPS`) | physics backend (`BACKEND`) |
|---|---|---|
| `wojtek` | 300M | `warp` |
| `go1`, `barkour`, `spot`, `go2` | 1B | `jax` (the Playground robots do not run on warp) |

Shared by all runs: 256 levels x 32 envs per iteration, 300-step levels (Spot 780),
`level_space=param`, `arena=benchmark`. Final eval (`conf/eval.yaml`): 2048
levels per random set, 4 episodes x 2 passes, 64 levels per suite cell, 300
steps (Spot 780), success = no fall and >= 1 m x robot scale, CVaR 10 / 25 %, seed
1234, on the training backend. Run names: `ued_<robot>_<mode>_seed<N>`,
evals `ued_<robot>_<mode>_seed<N>_final_eval`.

### 1. Cluster setup

- Linux x86_64, one NVIDIA GPU per run (H100 used), CUDA 12 driver.
- On the cluster: clone at the paper commit, `uv sync --locked --extra dev`,
  `wandb login`.
- Edit the cluster values at the top of `slurm/*.sbatch` (account,
  partition, log dir, `REPO`, `PDW`, wandb entity / project):
  [docs/slurm.md](docs/slurm.md#cluster-specific-values).

### 2. Assets and frozen copy

Jobs run from a frozen copy of `src conf assets`, so later edits do not
reach queued jobs. The build job creates the robot model, the arena and the
menagerie meshes inside it:

```bash
PDW=/path/to/work/dir                  # as PDW in slurm/*.sbatch
COPY="$PDW/copies/$(git rev-parse --short HEAD)"
mkdir -p "$PDW/logs" "$COPY"            # SLURM does not create the log dir
git archive HEAD src conf assets | tar -x -C "$COPY"
BUILD=$(SRC="$COPY/src" sbatch --parsable slurm/build_assets.sbatch)
```

### 3. Train (60 tasks per robot)

```bash
export SRC="$COPY/src" NSEEDS=5 TAGS=paper
export MODES="dr dr_easy dr_mixed plr sfl accel accel_grow accel_sfl paired paired_uni paired_bi paired_evo"

# Wojtek: 300M steps, warp; each robot's eval array starts task by task
# as its training task finishes (aftercorr)
T=$(ROBOT=wojtek NUM_TIMESTEPS=300000000 BACKEND=warp sbatch --parsable \
  --time=04:00:00 --dependency="afterok:$BUILD" --array=0-59 slurm/brax_ued.sbatch)
ROBOT=wojtek BACKEND=warp sbatch --dependency="aftercorr:$T" --array=0-59 \
  slurm/eval_brax.sbatch

# Playground robots: 1B steps, jax
for ROBOT in go1 barkour spot go2; do
  T=$(ROBOT="$ROBOT" NUM_TIMESTEPS=1000000000 BACKEND=jax sbatch --parsable \
    --time=06:00:00 --dependency="afterok:$BUILD" --array=0-59 slurm/brax_ued.sbatch)
  ROBOT="$ROBOT" BACKEND=jax sbatch --dependency="aftercorr:$T" --array=0-59 \
    slurm/eval_brax.sbatch
done
```

- WCSS caps submitted jobs per user (500 array tasks): the full grid is
  600 tasks, so submit the last robot once the first tasks have finished.

- Task = mode index x 5 + seed; checkpoints:
  `$PDW/artifacts/brax_ued/ued_<robot>_<mode>_seed<N>.pkl`.
- Variables and defaults: [docs/slurm.md](docs/slurm.md#job-variables).
- Cost: Wojtek ~8 GPU-h per seed over the 12 methods (warp, H100); a
  Playground run is ~1 GPU-h per 1B steps, PAIRED variants ~2x (two
  students). Throughput: [docs/training.md](docs/training.md#throughput).

### 4. Final evals

Submitted with the training arrays above (`aftercorr`). To re-run them for
finished checkpoints:

```bash
ROBOT=wojtek BACKEND=warp sbatch --array=0-59 slurm/eval_brax.sbatch
for ROBOT in go1 barkour spot go2; do
  ROBOT="$ROBOT" BACKEND=jax sbatch --array=0-59 slurm/eval_brax.sbatch
done
```

- Writes `$PDW/runs/final_eval/ued_<robot>_<mode>_seed<N>_final_eval.{json,npz}`
  and logs the metrics to a wandb run of that name.
- Optional transfer test (CPU MuJoCo physics variants):
  `slurm/eval_sim2sim.sbatch`, [docs/evaluation.md](docs/evaluation.md#sim2sim).

Copy the eval dir to the laptop (`PDW` = the cluster path); the rest runs
without a GPU:

```bash
rsync -a "user@login-node:$PDW/runs/final_eval/" artifacts/final_eval/
```

### 5. Domination comparison

```bash
just domination eval_dir=artifacts/final_eval                  # ACCEL vs DR
just domination eval_dir=artifacts/final_eval method_a=plr method_b=dr \
  out=artifacts/paper/domination_plr_dr
```

- Out: `artifacts/paper/domination/` (`domination.json`, heatmaps,
  `domination_matrix.pdf`). Method: [docs/results.md](docs/results.md#domination-comparison).

### 6. Tables and figures

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

- `dump_wandb` / `dump_history` read the wandb project set in
  `src/paper/runs.py` (`ENTITY`, `PROJECT`).
- A robot without runs: `prefixes.<robot>=null`.
- Details: [docs/results.md](docs/results.md).

### 7. Overleaf

The tables are bare booktabs `tabular`s: upload `artifacts/paper/tables/*.tex`
and the PDFs in `artifacts/paper/figs/` (into the project's `figs/`) to the
paper's Overleaf project; the paper owns floats, captions and labels.

## Train and export a policy

A deployable model for the real robot, next to the benchmark: define the
run in one YAML, train one policy, export it to ONNX.

```bash
cp conf/policy/template.yaml conf/policy/my_robot.yaml   # every value: robot, mode switches, terrain, PPO, network
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

## Documentation

Map of all pages: [docs/index.md](docs/index.md).

| Doc | Read it for |
|---|---|
| [overview](docs/overview.md) | methods, robots, level space, metrics, pipeline, repo map |
| [setup](docs/setup.md) | install, dependency pins, assets, outputs |
| [slurm](docs/slurm.md) | job variables, per-robot budgets, cluster values |
| [configuration](docs/configuration.md) | every Hydra config and field |
| [export](docs/export.md) | train from one YAML, export to ONNX, the deployment README |
| [development](docs/development.md) | layout rules, recipes, tests, adding features |
