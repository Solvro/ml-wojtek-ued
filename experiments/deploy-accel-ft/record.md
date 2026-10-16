# deploy-accel-ft — record

Code: `0fb3708` (jgilewicz/accel-ft), frozen copy
`$PDW/copies/dft-0fb3708` (`PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued`).
Upstream: `/home/jakgil6519/workspace/w01tek-e44c1a9`, base checkpoint
`training/runs/deploy_ws_upstream_1p5b_seed0/checkpoints/001203240960`
(305M A1-long + 1.2B continuation).

| date | job | what |
|---|---|---|
| 2026-10-16 10:40 | 6111593 | setup (CPU): build_model, accel + accel_deploy tiled arenas, import base → `artifacts/upstream/base_1p5b.pkl`, `just export` check |
| 2026-10-16 10:40 | 6111594 | base eval_brax (native + v10box), afterok 6111593 |
| 2026-10-16 10:40 | 6111595 | train array 0-8 (A accel, B accel lr 1e-4, C dr; seeds 0-2), 1B steps, afterok 6111593 |
| 2026-10-16 10:40 | 6111596 | eval_brax array 0-8, aftercorr 6111595 |
| 2026-10-16 ~11:05 | 6111595, 6111596 | cancelled: launch 1 runs reached 50-150M; A seed1 and all C seeds collapsed |

## Launch 1 finding: env blow-ups poison the obs normalizer

- `diverged`/`nonfinite_per_step` > 0 in 8 of 9 runs (warp; ~1-5 env steps
  per 2.46M-step iteration, more on DR's full-difficulty levels).
- Collapses (eval d1.0 success → ~0 within one iteration, reward per step
  negative, regret spike) coincide with those iterations: dr_seed0 it 1
  (mean forward_vel 96 m/s), dr_seed2 it 2/28, dr_seed1 and accel_seed1
  it 18. The NaN step itself was ended by the first guard (0fb3708), but
  the exploding-but-finite steps before it entered training.
- Checkpoints vs the imported base normalizer (count 1.5e9): obs std ratio
  max 1.04 (accel_seed0, healthy), 2045 (accel_seed1), 640753 (dr_seed1),
  all in joint-velocity dims. The count is so large that the poisoned
  variance never washes out.
- T6 `deploy_ws_accel_seed2` (jax, 140M): the same event ending in nan —
  iteration 55 healthy (reward 0.0515, train success 0.73, falls 8e-5),
  iteration 56 nan everywhere; no loss/regret precursor.
- Fix (relaunch code): `FunctionalEnv._end_diverged` also ends any step with
  |qvel| > 200; brax_ued stops if any obs std leaves [0.1, 10]x of the
  `--init_from` normalizer; PPO losses + grad norm logged.

## Launch 2 (code b84a645, runs `dft2_*`, 300M = the go/no-go)

Approved by the coordinator: ~18 GPU-h total incl. launch 1.

| date | job | what |
|---|---|---|
| 2026-10-16 11:30 | 6111933 | setup, frozen copy `$PDW/copies/dft-b84a645` |
| 2026-10-16 11:30 | 6111934 | base eval_brax (native + v10box) |
| 2026-10-16 11:30 | 6111935 | train array 0-8, 300M steps |
| 2026-10-16 11:30 | 6111936 | eval_brax array 0-8, aftercorr 6111935 |

## Results (launch 2, 300M)

- No arm beats the base (criterion above); no export. B: cmd +2.7 [+2.1,
  +3.4] passes, success +0.8 [+0.0, +1.6] fails. A worse everywhere. C:
  success +1.3, fall −9.3, cmd +0.1. Details: `report.md`, page
  https://claude.ai/code/artifact/8d375cee-c869-4b66-a942-8331da0f9d4d
- Cost: 10.6 GPU-h total (sacct).

## Launch 3 (code ce15a38, budget raised to ~25 GPU-h)

Coordinator: run all three proposals; flat guard on >=512 levels; winner
across ACCEL arms only (DR arms are controls).

| date | job | what |
|---|---|---|
| 2026-10-16 12:42 | 6112488 | setup, frozen copy `$PDW/copies/dft-ce15a38` |
| 2026-10-16 12:42 | 6112489 | train 0-11: accel_hard_lr1e-4, dr_lr1e-4, accel_sfl_lr1e-4 (300M from base), accel_lr1e-4_1b (+700M from launch 2's B) |
| 2026-10-16 12:42 | 6112490 | eval 0-11 (flat 1024 + native + v10box), aftercorr 6112489 |
| 2026-10-16 12:42 | 6112491 | flat-only 1024-level eval of launch 2's 9 policies + base |
