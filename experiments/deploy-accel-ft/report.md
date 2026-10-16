# deploy-accel-ft: ACCEL fine-tune vs the 1.5B base

Question: does an ACCEL fine-tune of the upstream 1.5B-step
`terrain_blind_v4_4d` policy (w01-tek trainer, imported) beat that base on
the deploy arena? Report page:
https://claude.ai/code/artifact/8d375cee-c869-4b66-a942-8331da0f9d4d

## Answer

- **Yes: F = ACCEL with SFL scoring (`--score learnability`, `--generator
  mixed --mutation grow`) at lr 1e-4, 300M steps**, pooled and in every seed,
  no guard violation. Exported: F seed 1.
- The gain is not ACCEL-specific: DR at lr 1e-4 (control) reaches the same
  primary numbers. Lower learning rate + hard training levels is what
  moved the policy; SFL gets ACCEL's buffer that hard.

## Setup

- Base: `deploy_ws_upstream_1p5b_seed0` checkpoint `001203240960`,
  re-imported with this code (tiled-arena sidecar, warp).
- All arms `--init_from` the base (B→1B: from B's 300M checkpoints), deploy
  profile, upstream rewards / DR / network, clip 0.3, warp tiled arena,
  seeds 0-2, `--n_edits 2 --edit_scale 0.1` for ACCEL:

  | arm | flags | steps |
  |---|---|---|
  | A | `--mode accel` (T6), lr 3e-4 | 300M |
  | B | A + `--learning_rate 1e-4` | 300M |
  | B→1B | B, continued from its 300M checkpoint (fresh Adam, fresh buffer) | +700M |
  | D | B + `--generator mixed --mutation grow` | 300M |
  | F | D + `--score learnability` (SFL) | 300M |
  | C | `--mode dr` (control), lr 3e-4 | 300M |
  | E | `--mode dr --learning_rate 1e-4` (control) | 300M |

- Eval: `eval_brax.py` defaults (2048 levels x 4 episodes x 2 passes) on
  warp, the same levels and reset keys for every policy; native, v10box
  (`--arena benchmark --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.14`), and
  `--flat_only --flat_levels 1024` for the flat guard.
- CIs (`analyze.py`): 2000 bootstrap draws over seeds x levels (base:
  levels); differences paired by level.
- Win criterion (fixed before running): both primary metrics
  (`random/d1.0/cmd_success_rate`, `random/d1.0/success_rate`) above the
  base with non-overlapping 95% CIs; no drop > 2 points on flat / d0.25;
  d1.0 fall rate not worse. Winner picked among ACCEL arms only.

## Results (native deploy eval, %, 3-seed pool [95% CI]; paired diff in points)

| policy | d1.0 cmd success | d1.0 success | flat (1024) | d0.25 | d1.0 fall | suite | verdict |
|---|---|---|---|---|---|---|---|
| base 1.5B | 41.6 [40.7, 42.5] | 32.1 [31.1, 33.1] | 73.3 | 70.9 | 30.3 | 47.7 | - |
| A ACCEL lr 3e-4 | 40.5 (−1.1) | 30.5 (−1.6) | −2.6 | −2.3 | −2.9 | 42.7 | no |
| B ACCEL lr 1e-4 | 44.4 (+2.7) | 32.9 (+0.8) | −2.8 | −1.8 | −0.4 | 45.6 | no (success, flat) |
| B→1B | 46.4 (+4.8 [+2.3, +6.5]) | 36.8 (+4.7 [+4.0, +5.3]) | −2.0 | −0.2 | −1.4 | 46.7 | pooled yes; seed 0 fails flat |
| D ACCEL harder | 47.6 (+5.9 [+5.0, +6.8]) | 36.9 (+4.8 [+3.3, +6.1]) | −2.3 | −1.2 | −6.2 | 47.6 | seeds 0, 1 yes; pooled fails flat |
| **F ACCEL SFL** | **47.9 (+6.3 [+5.7, +6.8])** | **38.8 (+6.7 [+5.9, +7.4])** | −0.7 | +0.4 | −4.9 | 49.0 | **yes, all seeds** |
| C DR lr 3e-4 | 41.7 (+0.1) | 33.4 (+1.3) | −0.9 | −1.6 | −9.3 | 44.8 | no (control) |
| E DR lr 1e-4 | 47.9 (+6.3) | 38.8 (+6.7) | −0.5 | +0.2 | −8.3 | 48.4 | yes (control only) |

- F per seed: d1.0 cmd 48.1 / 47.8 / 47.7, success 38.4 / 39.3 / 38.7.
- v10box d1.0 success: base 66.8; F 75.5 (+8.7), E 77.7, D 75.1, B→1B 71.6,
  B 68.5, C 73.6, A 63.8. cmd: base 51.9; F 63.2 (+11.4).

## Export (F seed 1)

- `just export`: numpy vs brax inference 1.35e-5, robot runtime vs env
  reference 2.57e-4. `policy_meta.json` equals the base export's except
  `run_name` / `checkpoint` (CONSUMED keys, network, runtime unchanged).
- `/Users/kuba/orca/workspaces/ml-wojtek-ued/accel-ft/artifacts/deploy_accel_ft/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy/`
  (checkpoint `.pkl` + sidecar beside it; WCSS:
  `$PDW/artifacts/deploy_accel_ft/`).
- Videos (20 s, 1280x720, arc command 0.3 m/s + 0.2 rad/s, no falls):
  - `/Users/kuba/orca/workspaces/ml-wojtek-ued/accel-ft/artifacts/videos/dft2_sfl_seed1_deploy/`:
    `dft2_deploy_accel_sfl_lr1e-4_seed1_d0.75_s5.mp4` (5.7 m),
    `..._d1.0_s17.mp4` (4.0 m), `..._d1.0_s23.mp4` (5.7 m), `demo.mp4`;
  - `/Users/kuba/orca/workspaces/ml-wojtek-ued/accel-ft/artifacts/videos/dft2_sfl_seed1_v10/dft2_deploy_accel_sfl_lr1e-4_seed1_d1.0_s17.mp4`
    (v10 arena, 2.8 m).

## Why plain ACCEL did not move terrain success

- Training levels, second half (replay iterations = buffer levels), genome
  distance from flat / training success: A 0.045 / 73%, B 0.050 / 72%,
  B→1B 0.076 / 68%, D 0.271 / 52%, F 0.252 / 49%, C and E 0.345 / 29-35%.
- The easy generator + 2 edits of ±0.1 keeps the buffer next to flat; the
  DR-mixed generator + grow mutations make it hard; SFL keeps the levels
  near p = 0.5.
- lr 3e-4 degrades the base first (A everywhere; C's command tracking);
  lr 1e-4 does not.

## T6 seed 2 collapse (and launch 1)

- MJX blows up on single envs (warp: ~1-5 env steps per 2.46M; 66-95% of
  iterations on hard levels, 7% on near-flat ones). The blown-up obs enter
  brax's running obs normalizer, whose warm-started count (1.5e9) never
  washes them out.
- T6 `deploy_ws_accel_seed2` (jax): T6's report places the nan in iteration
  55's PPO update. Correction: iteration 56 logged a normal fall rate
  (7.2e-5/step, then 0 from 57 on), so its rollout ran on finite params for
  most of the iteration and went nan inside it, which fits an env blow-up.
  No per-update losses were kept, so an update-side nan inside iteration 56
  is not excluded; PR #26's param guard and this env guard cover both.
- Launch 1 (`dft_*`, code 0fb3708) ended nan steps but not the exploding
  finite ones: A seed 1 and all C seeds collapsed within one iteration of a
  blow-up; obs std ratios vs the base normalizer x1.04 (healthy), x2045,
  x640753 (collapsed), all joint-velocity dims. Cancelled at 50-150M.
- Guard (launches 2-3): `FunctionalEnv._end_diverged` ends any step with a
  non-finite state or |qvel| > 200; brax_ued stops if an obs std leaves
  [0.1, 10]x of the `--init_from` one (observed 0.96-1.20). Training-only;
  `tests/deploy` passes unchanged.
- Code after merging PR #26 differs from the launch-2 copy only in
  brax_ued's training loop and `load_run`'s nan refusal, so the launch-2
  evals stand for the merged code.

## Cost

22.1 GPU-h (lem-gpu-short, H100; sacct): launch 1 3.3 (cancelled), launch 2
7.3, launch 3 11.4 (training 7.4, evals 3.9, flat re-evals 0.2). ~185k env
steps/s; 300M in 28 min, 700M in 62 min.

## Files

`experiment.toml`, `record.md`, `setup.sbatch` / `train.sbatch` (`ARMS`) /
`eval.sbatch` (`ARMS`, `FLAT_ONLY`), `analyze.py` (CIs, verdict →
`results.json`), `fetch_curves.py` (→ `cache/curves.json`), `report.py` +
`report_template.html` + `notes.html` → `report.html`. eval_brax outputs:
`$PDW/runs/deploy_accel_ft/dft2_*`.
