# SLURM

Job arrays behind the paper grid; the commands in order:
[README](../README.md#reproduce-the-paper).

## Scripts

| script | job | resources (as in the file) |
|---|---|---|
| `build_assets.sbatch` | robot model, `accel` arena, `mujoco_menagerie` into the frozen copy | CPU partition, 4 CPUs, 16 GB, 30 min |
| `brax_ued.sbatch` | one training run per array task (`scripts/brax_ued.py`) | GPU partition, 1 H100, 4 CPUs, 96 GB, 16 h, `--array=0-5` |
| `eval_brax.sbatch` | one final eval per task (`scripts/eval_brax.py`) | as training, 3 h |
| `eval_sim2sim.sbatch` | one sim2sim eval per task (`scripts/eval_sim2sim.py`) | CPU partition, 16 CPUs, 32 GB, 2 h |

## Task layout

- `MODES` = space-separated `conf/mode` options; task =
  mode index x `NSEEDS` + seed offset; seed = `SEED0` + offset. Array size =
  `NSEEDS` x number of modes (`--array=0-59` for 12 modes x 5 seeds).
- Run name `<PREFIX>_<ROBOT>_<mode>_seed<N>`; eval `<run>_final_eval`.
- One array per robot (`ROBOT`).

## Job variables

Read by `brax_ued.sbatch`, `eval_brax.sbatch`, `eval_sim2sim.sbatch` and
turned into Hydra overrides.

| variable | default | meaning | scripts |
|---|---|---|---|
| `MODES` | `dr accel` (train); required (evals) | `conf/mode` options; train also takes `<mode>-s<scale>` = `mode.generator_scale=<scale>` | all three |
| `NSEEDS` / `SEED0` | `3` / `0` | seeds per mode / first seed | all three |
| `ROBOT` | `wojtek` | `conf/robot` option | all three |
| `ROBOT_ARGS` | — | `robot.*` overrides, space-separated (`"robot.kp=50 robot.scale=2.0"`); a run's evals need the same | all three |
| `PREFIX` | `ued` | run-name prefix | all three |
| `SPACE` | `param` | `conf/level_space` option | train, `eval_brax` |
| `BACKEND` | `auto` | `auto` / `jax` / `warp` | train, `eval_brax` |
| `NUM_TIMESTEPS` | `300000000` | env steps (`num_timesteps`) | train |
| `TAGS` | — | extra wandb tags, space-separated | train, `eval_brax` |
| `SRC` | `$REPO/src` | `src/` of a frozen copy with built `assets/` and `conf/` beside it | all four |
| `REPO` | `$HOME/workspace/ml-wojtek-ued` | clone with the synced `.venv` | all three |
| `WB_PROJECT` | `wojtek-benchmark-iclr` | wandb project | all three |
| `EVAL_DIR` | `runs/final_eval` | eval output dir under `PDW` | `eval_brax` |

## Per-robot budgets

| robot | `NUM_TIMESTEPS` | `BACKEND` | why |
|---|---|---|---|
| `wojtek` | `300000000` | `warp` (`auto` resolves to it on a GPU) | the default; 263k env steps/s ([training.md](training.md#throughput)) |
| `go1`, `go2`, `barkour`, `spot` | `1000000000` | `jax` | foot contacts diverge on warp ([robots.md](robots.md)); `backend=warp` is refused |

- Evals use the backend of the training run (`BACKEND` as in training).
- Compare runs only within one backend
  ([training.md](training.md#throughput)).

## Frozen copy

- Jobs run `$SRC/scripts/...` from a copy of `src conf assets` made with
  `git archive`, so edits after submission do not reach queued jobs.
- `conf/` and `assets/` are resolved from `SRC/..`
  ([configuration.md](configuration.md)).
- Assets are built once by `build_assets.sbatch`; the array tasks start
  together and must not rebuild (they depend on the build job).
- `REPO` holds the clone with the synced `.venv` (`uv run --no-sync`).

## Cluster-specific values

Set to WCSS (Lem, H100) at the top of every `slurm/*.sbatch`; another cluster
changes:

| where | value |
|---|---|
| `#SBATCH --account` | compute grant |
| `#SBATCH --partition`, `--gres` | GPU partition / GPU type (`build_assets`, `eval_sim2sim`: a CPU partition) |
| `#SBATCH --output`, `--error` | log directory (SLURM does not create it) |
| `REPO` | clone with the synced `.venv` (default; job variable) |
| `PDW` | work dir: checkpoints `artifacts/brax_ued/`, evals `runs/final_eval/`, `runs/sim2sim/` |
| `WB_ENTITY`, `WB_PROJECT` | wandb entity / project (`WB_PROJECT` default) (also `ENTITY`, `PROJECT` in `src/paper/runs.py`) |

## Environment the scripts export

- `JAX_PLATFORMS=cuda` on GPU jobs: fail instead of a silent CPU fallback;
  `cpu` on CPU jobs.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (warp,
  [environment.md](environment.md#why-the-arena-swap-is-free)).
- `PYTHONPATH=$SRC`, `HYDRA_FULL_ERROR=1`, `PYTHONNOUSERSITE=1`.
