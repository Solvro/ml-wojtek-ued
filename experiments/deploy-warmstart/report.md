# deploy-warmstart: upstream 1.5B policy + ACCEL fine-tune, exported

- **Date:** 2026-10-16. **Code:** `jgilewicz/deploy-warmstart` (frozen copy
  `deploy-warmstart-9099f43`), upstream w01-tek `e44c1a9`. **wandb:**
  `wojtyla`, tag `deploy-warmstart`.
- **Cost:** 19.3 GPU-h (budget 22): upstream continuation 2.5, ACCEL
  3 x 4.06 = 12.2, evals 4.2, the cancelled 305M warm start 0.4.
- **Jobs:** 6109841 (upstream 305M → +1.2B), 6109842 (import + export +
  v10 arena), 6109843_[0-2] (ACCEL), 6109844 / 6109845 (evals), 6109816
  (305M baseline eval), 6109817 (305M warm start, cancelled at 8 min).

## Verdict

- **Exported: `deploy_ws_accel_seed0`** (best of the 3 by d1.0 success +
  cmd_success, native 0.288 + 0.319, v10box 0.578 + 0.340).
- **ACCEL fine-tuning did not beat the imported upstream policy.** Against
  upstream 1.5B (no fine-tune), seed 0 trades success and command tracking
  for fewer falls: d1.0 success 0.288 vs 0.296, cmd_success 0.319 vs 0.380,
  fall rate 0.232 vs 0.322 (native); on v10box 0.578 vs 0.630, 0.340 vs
  0.445, falls 0.188 vs 0.292. Both are deployable; which to put on the
  robot is a call between "falls less" (seed 0) and "tracks commands
  better" (upstream 1.5B). Both export dirs are below.
- **Warm start held:** falls/step ~1e-4 for the whole 150M (go/no-go at
  ~30M: 1e-4, far below 0.05), train success 0.71-0.76. Seed 2 diverged to
  NaN at 140.1M (reward, travel, regret NaN from that iteration on); its
  final checkpoint is unusable and it is excluded.
- **The deploy numbers are low because the deploy eval is harder, not the
  policy:** on the v10 arena with a forward command inside both command
  boxes (v10box), upstream 1.5B scores d1.0 0.630 vs v10 seed4's 0.606,
  suite 0.751 vs 0.695. The native deploy eval draws stand / spin / strafe
  / backward commands and uses the 12 x 12 m arena.
- **Upstream 305M → 1.5B changed nothing measurable:** eval episode length
  stayed 897-959 (of 1000), eval reward 50-67, base-down ends 0-4 %.

## Native deploy eval

Deploy arena (12 x 12 m), deploy command distribution, `eval_brax.py` 512
levels x 2 passes x 4 episodes, 16 suite instances, 300-step levels.
`cmd` = cmd_success_rate (no fall and the command tracked).

| policy | d0.25 success | d0.25 cmd | d0.25 fall | d1.0 success | d1.0 cmd | d1.0 fall | d1.0 travel m | uniform success | suite success | flat success |
|---|---|---|---|---|---|---|---|---|---|---|
| upstream 305M, imported (no fine-tune) | 0.727 | 0.768 | 0.005 | 0.285 | 0.339 | 0.256 | 0.802 | 0.274 | 0.450 | 0.797 |
| upstream 1.5B, imported (no fine-tune) | 0.701 | 0.828 | 0.022 | 0.296 | 0.380 | 0.322 | 0.791 | 0.288 | 0.446 | 0.750 |
| **1.5B + ACCEL 150M, seed 0 (exported)** | 0.724 | 0.755 | 0.009 | 0.288 | 0.319 | 0.232 | 0.810 | 0.284 | 0.420 | 0.789 |
| 1.5B + ACCEL 150M, seed 1 | 0.716 | 0.757 | 0.010 | 0.280 | 0.326 | 0.243 | 0.801 | 0.287 | 0.417 | 0.758 |
| 1.5B + ACCEL 150M, seed 2 | NaN at 140M | | | | | | | | | |

## Apples-to-apples with v10 (v10box)

Same eval sets and level seeds as v10's final eval (`--seed 1234`), v10
6 x 6 m arena geometry (`--arena benchmark`), one uniform forward command
inside both command boxes: vx [0.2, 0.6], vy [-0.2, 0.2], wz [-0.4, 0.4],
height 0.14 (`--command_box`; eval-only, the command is a policy input, the
network is unchanged). First row: v10's own final eval as in the paper
(v10 command mix, 2048 levels).

| policy | d0.25 success | d0.25 cmd | d0.25 fall | d1.0 success | d1.0 cmd | d1.0 fall | d1.0 travel m | uniform success | suite success | flat success |
|---|---|---|---|---|---|---|---|---|---|---|
| v10 seed4, paper eval | 0.909 | - | 0.016 | 0.423 | - | 0.358 | 1.077 | 0.412 | 0.584 | 0.936 |
| v10 seed4 | 0.998 | 0.998 | 0.001 | 0.606 | 0.392 | 0.206 | 1.332 | 0.599 | 0.695 | 1.000 |
| upstream 1.5B, imported | 0.997 | 0.993 | 0.001 | 0.630 | 0.445 | 0.292 | 1.288 | 0.596 | 0.751 | 1.000 |
| **1.5B + ACCEL, seed 0 (exported)** | 0.990 | 0.981 | 0.000 | 0.578 | 0.340 | 0.188 | 1.259 | 0.575 | 0.693 | 1.000 |
| 1.5B + ACCEL, seed 1 | 0.993 | 0.983 | 0.000 | 0.565 | 0.344 | 0.181 | 1.196 | 0.559 | 0.681 | 1.000 |

## Training curves (in-loop eval, 2 x 512 levels)

| seed | env steps | falls / step | train success | eval d0.25 fall | eval d1.0 success | eval d1.0 fall | steps/s |
|---|---|---|---|---|---|---|---|
| 0 | 2.5M | 9.4e-5 | 0.721 | 0.016 | 0.298 | 0.333 | 5.8k |
| 0 | 32M | 1.0e-4 | 0.730 | 0.010 | 0.321 | 0.262 | 9.4k |
| 0 | 81M | 8.0e-5 | 0.755 | 0.016 | 0.270 | 0.286 | 10.4k |
| 0 | 152M | 1.1e-4 | 0.708 | 0.008 | 0.280 | 0.245 | 10.7k |
| 1 | 2.5M | 9.1e-5 | 0.716 | 0.022 | 0.298 | 0.338 | 5.8k |
| 1 | 32M | 7.2e-5 | 0.750 | 0.012 | 0.310 | 0.293 | 9.4k |
| 1 | 152M | 8.4e-5 | 0.757 | 0.006 | 0.296 | 0.246 | 10.7k |
| 2 | 128M | 1.1e-4 | 0.745 | - | - | - | 10.7k |
| 2 | 140M+ | NaN | 0 | - | - | - | - |

## Upstream continuation, 305M → 1.5B (w01-tek train.py, restore)

Brax counts steps from 0 after a restore, so `env steps` below are on top
of A1-long's 305M. Eval = brax evaluator, 128 envs x 1000-step episodes.

| extra env steps | eval ep_len | eval ends base-down | eval no-progress cut | eval reward | train ep_len | steps/s |
|---|---|---|---|---|---|---|
| 0 | 919.8 | 0 | 0.156 | 57.9 | 226 | - |
| 100M | 948.6 | 0 | 0.117 | 60.1 | 633 | 126k |
| 300M | 946.3 | 0.008 | 0.117 | 61.7 | 800 | 136k |
| 501M | 958.7 | 0.016 | 0.070 | 61.2 | 799 | 138k |
| 702M | 926.3 | 0.008 | 0.133 | 57.6 | 761 | 141k |
| 902M | 945.4 | 0 | 0.125 | 62.6 | 894 | 142k |
| 1103M | 897.2 | 0.039 | 0.164 | 59.8 | 695 | 141k |
| 1203M | 917.3 | 0.031 | 0.172 | 64.3 | 688 | 141k |

## Converter and warm start

- `just import-upstream` on the 305M and 1.5B checkpoints: every check
  passed (CONSUMED env keys = upstream run.json, actor 46 / critic 78
  obs, action 12, network kwargs, policy and value shapes, normalizer
  keys). For 305M, our `policy.npz` is byte-identical to upstream's own
  `export_policy.py` output; `policy_meta.json` differs only in
  `checkpoint` / `run_name`.
- `just export` validations (numpy vs brax / robot runtime vs env):
  upstream 1.5B 2.1e-5 / 1.0e-5; seed 0 6.7e-5 / 1.3e-5 (bounds 1e-4 /
  1e-3).
- `--init_from` loads normalizer, policy and value, fresh optimizer; the
  first iteration already walks (falls/step 9e-5, train success 0.72).

## Export dirs (policy.npz + policy_meta.json, schema 2)

- Fine-tuned seed 0:
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/deploy_ws/deploy_ws_accel_seed0_deploy/`
- Upstream 1.5B, no fine-tune:
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/upstream/a1xl_deploy/`
- Checkpoints + sidecars beside them (`deploy_ws_accel_seed0.pkl/.json`,
  `a1xl.pkl/.json`); on WCSS under `$PDW/artifacts/deploy_ws/` and
  `~/workspace/deploy-warmstart-9099f43/artifacts/upstream/`.

## Videos (seed 0, `render_video.py`, 20 s, 1280 x 720)

Deploy arena, held forward arc vx 0.4 m/s, wz 0.2 rad/s; no fall in any
clip. Frame PNGs at t = 10 s beside each mp4.

- d0.25 seed 3, 5.72 m from spawn:
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/videos/deploy/deploy_ws_accel_seed0_d0.25_s3.mp4`
- d0.5 seed 7, 5.78 m:
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/videos/deploy/deploy_ws_accel_seed0_d0.5_s7.mp4`
- d1.0 seed 17, 4.01 m:
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/videos/deploy/deploy_ws_accel_seed0_d1.0_s17.mp4`
- All three: `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/videos/deploy/demo.mp4`
- v10 arena (`--arena benchmark`), level 1.0:17, the v10 demo's command
  (vx 0.3, wz 0.2): no fall, 1.83 m from spawn (v10 seed4's demo: 2.89 m):
  `/Users/kuba/orca/workspaces/ml-wojtek-ued/deploy-ab/artifacts/videos/deploy/v10_arena/deploy_ws_accel_seed0_d1.0_s17.mp4`

## Running it on the robot ([docs/deploy.md](../../docs/deploy.md#run))

1. Sim first, from `./dev.sh` in w01-tek:
   `ros2 launch wojtek_pc sim.launch.py policy:=<export dir>`.
2. Robot: `ros2 launch wojtek_bringup robot.launch.py policy:=<dir on the
   RPi>`; the contract's `pd` block (kp 40, kd 1.6, max_torque 32) feeds
   the servo xacro; zero, stand up, arm by hand.
3. The actor reads the IMU (`gyro`, `gravity`): the CANdle-hat BMI must be
   wired into the policy node first (upstream's preset note).

## Caveats

- One seed of the upstream run; 2 usable fine-tune seeds.
- The v10 arena clip and v10box eval put a deploy-trained policy on a
  smaller arena it never trained on; both are eval-only.

- Report page: https://claude.ai/code/artifact/cc0573b2-d8f2-47e2-b5af-855e6cfc64ef
