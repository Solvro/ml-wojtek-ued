# v2-dr-terrain-validity: does PPO + DR terrain learn on the v2 fixed hfield?

- **Status:** running
- **Account:** `hpc-danbor2008-1756464546`
- **Started:** 2026-10-07
- **Code:** branch `debug/eval-terrain-difficulty`, commit `a24d8e4` + 12 uncommitted files on the WCSS checkout

## Question

- Sanity baseline after the hfield-batching fix (`3415ceb`): does
  `ppo_dr_terrain.py` learn a stable gait on the randomized v2 terrain?
- Success: rising train reward, falling `fall_toggle_rate`, eval at hfield
  difficulty 0.0 / 0.5 / 1.0 degrades gracefully with difficulty.

## Method

- Train: `src/scripts/ppo_dr_terrain.py`, 4096 envs, 400 steps/episode,
  400 iterations, seed 1234, γ=0.99, checkpoint + eval every 25 iterations.
- Hfield: amplitude 0.3, coarse_n 13, pad_radius 0.55, pad_taper 0.2.
- Eval: `src/scripts/eval_levels.py`, 2000 levels × difficulty {0.0, 0.5, 1.0}.
- Job scripts: `$PDW/jobs/{v2_dr_train,v2_eval_levels}.sbatch`, `common.sh`.
- Outputs: `$PDW/artifacts/ppo_dr_terrain/v2_ppo_dr_terrain_validity_seed1234/`,
  `$PDW/runs/`, `$PDW/logs/`.
- wandb: `wojtyla`, run `pg8njbgi`, tags `[v2-fixed-hfield, validity]`.

## Budget

| | |
|---|---|
| resources | 1× H100, 4 CPU, 64 GB |
| wall time req | train 20 h, eval 2.5 h |
| expected use | train ~14.6 h (~2.15 min/iter), eval ≤2.5 h |
| **estimated GPU-hours** | ~17 |

## Runs

| Job ID | Date | Partition | Cores | Mem | Wall req/used | MaxRSS | CPUh | Exit | Output | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| 6051886 | 2026-10-07 | lem-gpu-short | 4 | 64G | 20:00 / running | | | | logs/wued-v2-dr-train_6051886.out | a24d8e4 (dirty) |
| 6051887 | 2026-10-07 | lem-gpu-short | 4 | 64G | 2:30 / pending (afterany:6051886) | | | | logs/wued-v2-eval_6051887.out | a24d8e4 (dirty) |

## Progress notes

- 2026-10-08 09:26 — train at iter 273/400; mean_reward 12.3 (from −38.7);
  fall_toggle_rate 0.40 (peak 0.93 @ iter 75); GPU 100 %, 72 GB.

## Results
