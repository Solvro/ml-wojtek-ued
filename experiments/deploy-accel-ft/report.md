# deploy-accel-ft: ACCEL fine-tune vs the 1.5B base

Question: does an ACCEL fine-tune of the upstream 1.5B-step
`terrain_blind_v4_4d` policy (w01-tek trainer, imported) beat that base on
the deploy arena? Report page:
https://claude.ai/code/artifact/8d375cee-c869-4b66-a942-8331da0f9d4d

## Setup

- Base: `deploy_ws_upstream_1p5b_seed0` checkpoint `001203240960`,
  re-imported with this code (tiled-arena sidecar, warp).
- Arms, `--init_from` the base, deploy profile, upstream rewards / DR /
  network, clip 0.3, warp tiled arena, 300M env steps (the go/no-go point),
  seeds 0-2: A `--mode accel` (n_edits 2, edit_scale 0.1, lr 3e-4 = T6);
  B A + `--learning_rate 1e-4`; C `--mode dr` (uniform levels, lr 3e-4).
- Eval: `eval_brax.py` defaults (2048 levels x 4 episodes x 2 passes), warp,
  same levels and reset keys for every policy; native and v10box
  (`--arena benchmark --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.14`).
- CIs (`analyze.py`): 2000 bootstrap draws over seeds x levels (base:
  levels); differences paired by level.
- Win criterion (fixed before running): both primary metrics
  (`random/d1.0/cmd_success_rate`, `random/d1.0/success_rate`) above the
  base with non-overlapping 95% CIs, no drop > 2 points on flat / d0.25,
  d1.0 fall rate not worse.

## Results (native deploy eval, %, 95% CI; diff vs base in points)

| policy | d1.0 cmd success | d1.0 success | d1.0 fall | d0.25 success | flat success | suite success |
|---|---|---|---|---|---|---|
| base 1.5B | 41.6 [40.7, 42.5] | 32.1 [31.1, 33.1] | 30.3 | 70.9 | 75.0 [70.9, 79.1] | 47.7 |
| A ACCEL | 40.5 (−1.1 [−1.6, −0.5]) | 30.5 (−1.6 [−2.2, −0.9]) | 27.4 | 68.6 | 70.8 | 42.7 |
| B ACCEL lr 1e-4 | **44.4** (+2.7 [+2.1, +3.4]) | 32.9 (+0.8 [+0.0, +1.6]) | 29.9 | 69.1 | 70.2 | 45.6 |
| C DR | 41.7 (+0.1 [−0.6, +0.7]) | 33.4 (+1.3 [+0.6, +2.0]) | 21.1 | 69.3 | 72.9 | 44.8 |

v10box (v10 arena, forward commands): d1.0 success base 66.8, A 63.8, B
68.5 (+1.7 [+1.0, +2.5]), C 73.6 (+6.8 [+5.9, +7.7]); cmd base 51.9, A
46.8, B 54.2, C 54.8.

## Verdict

- **No arm beats the base**, per seed or pooled. No export; the policy to
  upload stays the base.
- B passes the cmd-success primary in every seed (44.0-44.8 vs 41.6, CIs
  apart) and fails d1.0 success (+0.8, CIs overlap). A is worse than the
  base on every set (lr 3e-4 is too high for this fine-tune). C improves
  terrain robustness (fall −9.3, v10box success +6.8) but not tracking.
- Flat guard fails for every arm (−2 to −5 points), inside the flat set's
  own ±4-point CI (64 levels); d0.25 (2048 levels) drops 1.6-2.3 points.
- Why ACCEL does not move terrain success: its buffer stays next to flat
  (genome distance 0.045-0.050 vs DR 0.345; training success 0.72-0.73 vs
  DR 0.29 and d1.0 eval 0.32).

## T6 seed 2 collapse (and launch 1)

- MJX blows up on single envs (warp: ~1-5 env steps per 2.46M; 92% of DR
  iterations, 7% of ACCEL ones). The blown-up obs enter brax's running obs
  normalizer, whose warm-started count (1.5e9) never washes them out.
- T6 `deploy_ws_accel_seed2` (jax): healthy at iteration 55, nan at 56.
  T6's report places the nan in iteration 55's PPO update; but iteration 56
  logged a normal fall rate (7.2e-5/step, then 0 from 57 on), so its
  rollout ran on finite params for most of the iteration and went nan
  inside it: consistent with an env blow-up, not with params nan from its
  start. No per-update losses were kept, so an update-side nan inside
  iteration 56 is not excluded; PR #26's param guard and this env guard
  cover both.
- Launch 1 (`dft_*`, code 0fb3708) ended nan steps but not the exploding
  finite ones: A seed 1 and all C seeds collapsed within one iteration of a
  blow-up; obs std ratios vs the base normalizer x1.04 (healthy), x2045,
  x640753 (collapsed), all joint-velocity dims. Cancelled at 50-150M.
- Guard (launch 2, `dft2_*`, b84a645): `FunctionalEnv._end_diverged` ends
  any step with a non-finite state or |qvel| > 200; brax_ued stops if an
  obs std leaves [0.1, 10]x of the `--init_from` one (launch 2: 0.96-1.20).
  Training-only; `tests/deploy` passes unchanged.
- Code after merging PR #26 differs from the launch-2 copy only in
  brax_ued's training loop and `load_run`'s nan refusal, so the launch-2
  evals stand for the merged code.

## Proposals (measured: 0.47 GPU-h train + 0.31 GPU-h evals per 300M run)

1. Harder ACCEL at lr 1e-4 (`--generator mixed --mutation grow`) plus DR at
   lr 1e-4 (removes C's learning-rate confound): 6 runs, ~4.7 GPU-h.
   Recommended first.
2. Proposal 1 with `--score learnability` (SFL): 3 runs, ~2.4 GPU-h.
3. Continue B to 1B (`--init_from` its 300M checkpoints, +700M): 3 runs,
   ~3.6 GPU-h.

## Cost

10.6 GPU-h (lem-gpu-short, H100): launch 1 3.3 (cancelled), launch 2 7.3
(training 4.29, evals 3.05). ~185k env steps/s; 300M in 28 min.

## Files

`experiment.toml`, `record.md`, `setup.sbatch` / `train.sbatch` /
`eval.sbatch`, `analyze.py` (CIs, verdict → `results.json`),
`fetch_curves.py`, `report.py` + `report_template.html` + `notes.html` →
`report.html`. eval_brax outputs: `$PDW/runs/deploy_accel_ft/dft2_*`.
