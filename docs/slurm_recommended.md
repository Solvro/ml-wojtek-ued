# Slurm — recommended commands

Reference invocations for `slurm.sh` (see the script for the full env setup —
venv, `HYDRA_FULL_ERROR`, `PYTHONPATH`, arena build gating). All three train
commands below target the same env budget (~8192 parallel envs) and seed, so
they're comparable apples-to-apples.

`ALGO` selects the train script (`ppo` | `ppo_dr` | `accel` | `accel_top_k`);
`TRAIN_ARGS` are Hydra overrides appended to it. `MODE=eval` switches the
script to `scripts/eval_levels.py` with `EVAL_ARGS` instead, and always forces
the `accel` arena build regardless of `ALGO`.

**Terrain difficulty (`HFIELD_ARGS`)** — the physical elevation envelope
(`amplitude`, `patch_radius`, ...) is baked once into the compiled arena's
MJCF `hfield` `size`/`pos` by `hfield.arena` at build time; it is **not**
reactive to `hfield_params.*` overrides passed only to `TRAIN_ARGS`/
`EVAL_ARGS`, since `hfield_size` is static post-compile (see
[ued-accel.md](ued-accel.md#arena)). Use `HFIELD_ARGS` to set
`hfield_params.*` overrides once — `slurm.sh` passes it to the arena build
*and* appends it to `EVAL_ARGS` (always) and `TRAIN_ARGS` (for
`accel`/`accel_top_k` only, since `ppo`/`ppo_dr` configs have no
`hfield_params` field), so the compiled arena and the runtime `HFieldManager`
never drift apart:

```bash
sbatch --job-name=eval_ppo_dr --export=ALL,MODE=eval,ALGO=ppo_dr,HFIELD_ARGS="hfield_params.amplitude=0.30 hfield_params.coarse_n=13",EVAL_ARGS="run=ppo_dr_seed_1234_envnum_8192" slurm.sh
```

## Training

**ACCEL (vanilla, single level per iteration)** — `environment_params.envs_num`
directly sets the rollout width here.

```bash
sbatch --job-name=accel --export=ALL,ALGO=accel,TRAIN_ARGS="environment_params.envs_num=8192 accel_params.max_iterations=20000 accel_params.min_iterations=5000 accel_params.plateau_patience=1000 accel_params.plateau_rel_tol=0.001 wandb_params.name=accel_seed_1234_envnum_8192 environment_params.seed=1234 training_params.save_every=1000" slurm.sh
```

**ACCEL top-k (batched lanes)** — `environment_params.envs_num` is **not**
read by `scripts/accel_top_k.py`; the rollout width is
`accel_params.num_lanes * accel_params.envs_per_lane` instead
(128 × 64 = 8192, matching vanilla ACCEL's budget). `top_k=32` keeps the
default 25% curation ratio (`16/64`). Each iteration scores more candidates
and pools a bigger batch into one PPO update than vanilla, so convergence
should need fewer iterations — `max_iterations` is set lower accordingly.

```bash
sbatch --job-name=accel_top_k --export=ALL,ALGO=accel_top_k,TRAIN_ARGS="accel_params.num_lanes=128 accel_params.envs_per_lane=64 accel_params.top_k=32 accel_params.max_iterations=10000 accel_params.min_iterations=2500 accel_params.plateau_patience=500 accel_params.plateau_rel_tol=0.001 wandb_params.name=accel_top_k_seed_1234_envnum_8192 environment_params.seed=1234 training_params.save_every=1000" slurm.sh
```

**PPO + domain randomization** — `scripts/ppo_dr.py` has no `PlateauStopper`
(fixed `training_params.iterations` loop), so "convergence" here is bounded by
the `lem-gpu-short` partition's 3h wall-clock limit, not the iteration count.
The large iteration cap is intentional (matches `slurm.sh`'s own default) —
rely on `save_every` for a usable checkpoint whenever the job gets killed.
`scripts/ppo_dr.py` does **not** restore from a checkpoint on start, so a
follow-up `sbatch` after timeout restarts training from scratch unless you add
resume logic yourself.

```bash
sbatch --job-name=ppo_dr --export=ALL,ALGO=ppo_dr,TRAIN_ARGS="environment_params.envs_num=8192 training_params.iterations=5000000 wandb_params.name=ppo_dr_seed_1234_envnum_8192 environment_params.seed=1234 training_params.save_every=1000" slurm.sh
```

Chain an eval job to run right after training finishes with
`sbatch --dependency=afterok:<TRAIN_JOBID> ...`.

## Evaluation

`run=` must match the training run's `wandb_params.name` (it's also the
checkpoint directory name, `artifacts/<algo>/<run>/`) — otherwise eval falls
back to `latest_run_dir`, the newest directory by mtime under
`artifacts/<algo>/`, which is fragile once more than one run per algo exists.

**Generalization eval** (`scripts/eval_levels.py`, via `MODE=eval` —
freshly generated heightfield levels, logs `mean_reward`/`success_rate`/CVaR):

```bash
sbatch --job-name=eval_accel --export=ALL,MODE=eval,ALGO=accel,EVAL_ARGS="run=accel_seed_1234_envnum_8192 wandb_params.name=accel_seed_1234_envnum_8192_eval_levels" slurm.sh

sbatch --job-name=eval_accel_top_k --export=ALL,MODE=eval,ALGO=accel_top_k,EVAL_ARGS="run=accel_top_k_seed_1234_envnum_8192 wandb_params.name=accel_top_k_seed_1234_envnum_8192_eval_levels" slurm.sh

sbatch --job-name=eval_ppo_dr --export=ALL,MODE=eval,ALGO=ppo_dr,EVAL_ARGS="run=ppo_dr_seed_1234_envnum_8192 wandb_params.name=ppo_dr_seed_1234_envnum_8192_eval_levels" slurm.sh
```

Defaults (`num_levels=10000`, `episodes_per_level=8`, `chunk_size=256`,
`cvar_alpha=0.25`) are already H100-scale ([configuration.md](configuration.md)
has the full field list); add `step=N` to `EVAL_ARGS` to pin a specific
checkpoint instead of the latest (`max_to_keep=5` — only the last 5 survive).

**Quick fixed-course sanity check** (`scripts/eval.py`, one `mean_return`, no
arena) — not wired into `slurm.sh`'s `MODE=`, run it directly on a GPU node
instead of through `sbatch`:

```bash
just eval algo=accel run=accel_seed_1234_envnum_8192
just eval algo=accel_top_k run=accel_top_k_seed_1234_envnum_8192
just eval algo=ppo_dr run=ppo_dr_seed_1234_envnum_8192
```
