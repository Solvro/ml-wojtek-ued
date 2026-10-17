# robot-ft: ACCEL fine-tune on the robot's contract

Question: does T8's winning recipe (ACCEL, SFL scoring, mixed generator,
grow mutation, lr 1e-4, 300M) beat its base when the base is trained by
upstream's own `train.py` on the contract of the policy the robot runs?

## Answer

- **No.** Neither arm beats the base on any primary metric, pooled or in
  any seed; both fail the flat guard. **No export**;
  [Current robot policy](../../docs/deploy.md#current-policy) is unchanged.
- The base hardly falls on terrain (d1.0 fall rate 1.1%) but stalls
  (d1.0 mean travel 0.61 m in 6 s). Both fine-tunes move even less: ACCEL
  slightly (d1.0 success −2.3 points), DR a lot (d1.0 success 14.6% →
  0.9%). In upstream's battery they read as quieter (energy −8 to −34%)
  because they move less, not because they walk better.

## Target contract

- The brief named the policy `policy_source.py` pins
  (`wojtek-quiet-locomotion@553795b`, kp 20). The coordinator moved the
  target, by user decision, to the policy the RPi drop-in loads:
  `wojtek_flat_quiet_baseacc_s0`. The drop-in's `policy:=/home/rpi/policy`
  overrides the pin, so the robot team has to make sure the policy in
  `/home/rpi/policy` is the one they mean.
- Profile `robot` = resolved `flat_quiet_v6` + `pd_kp=40`, `pd_kd=0.8`;
  base recipe adds the `base_accel` dose (`brax_ued.UPSTREAM_RUNS["robot"]`).
  Established vs inferred keys: [docs/deploy.md](../../docs/deploy.md#robot-profile).
  The inferred ones (action scale, command box, joint limits, ctrl_dt,
  real_pose_ref) have to be checked against the deployed
  `policy_meta.json` before any export of this profile is loaded.
- Base export (setup job): `policy_meta.json` equals the resolved
  upstream contract (`tests/deploy/upstream_robot.json`) in every field but
  `run_name` / `checkpoint`; numpy vs brax 0.0, robot runtime vs env
  reference 7.1e-6.

## Setup

- Base: upstream `train.py` with `UPSTREAM_RUNS["robot"]`, from scratch, 2B
  steps, 8192 envs, warp (`rft_upstream_seed0`, 73 min); imported with the
  robot profile.
- Arms (`--init_from` the base, robot profile, clip 0.3, warp tiled
  deploy-geometry arena, seeds 0-2, 300M): ACCEL SFL lr 1e-4 (`--mode accel
  --score learnability --generator mixed --mutation grow --n_edits 2
  --edit_scale 0.1`), DR lr 1e-4 (control). No RND (upstream trains with it).
- Eval: `eval_brax.py` defaults on warp, same levels and keys for every
  policy: native, flat on 1024 levels, v10box (`--arena benchmark
  --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.125`). Upstream battery:
  `run.sh report` on upstream's flat scene; checkpoints converted with
  `to_upstream_run.py` (the converted base reproduces the upstream run's
  own report exactly).
- Criterion: `experiment.toml`, fixed before launch; `analyze.py` →
  `results.json`.

## Results (native eval, %, 3-seed pool; paired diff vs base [95% CI])

| policy | d1.0 cmd success | d1.0 success | d0.25 success | d1.0 fall | flat success | flat cmd success | suite | verdict |
|---|---|---|---|---|---|---|---|---|
| base (2B) | 56.9 [56.1, 57.7] | 14.6 [13.9, 15.3] | 37.6 | 1.1 | 42.5 | 99.6 | 23.3 | - |
| ACCEL SFL lr 1e-4 | 55.8 (−1.2 [−2.0, −0.1]) | 12.3 (−2.3 [−5.8, 0.0]) | −1.9 | −0.2 | −2.6 [−5.8, −0.5] | −8.5 [−12.2, −3.9] | 22.2 | no |
| DR lr 1e-4 (control) | 52.5 (−4.4) | 0.9 (−13.7) | −24.0 | −0.9 | −15.7 | −21.0 | 6.6 | no |

- ACCEL per seed (d1.0 cmd / success): 56.9 / 14.6, 54.9 / 8.6, 55.6 / 13.6.
- Success needs ≥ 1 m of travel; the robot command mix draws stand, spin
  and slow commands, so flat success is 42.5% while flat command tracking
  is 99.6%.
- v10box d1.0 success: base 32.8; ACCEL −8.4, DR −29.0 points.
- Upstream battery (flat scene; energy = mean total motor power, impact =
  peak base vertical accel; no falls anywhere):

  | policy | energy W | impact m/s² | vibration | spin err rad/s |
  |---|---|---|---|---|
  | base | 21.35 | 19.98 | 0.434 | 0.12 |
  | ACCEL seeds 0/1/2 | 18.8 / 18.0 / 19.7 | 19.0 / 16.6 / 17.7 | 0.40 | 0.18 / 0.30 / 0.30 |
  | DR seeds 0/1/2 | 14.7 / 14.4 / 14.0 | 13.2 / 11.8 / 17.5 | 0.37 | 0.22 / 0.24 / 0.24 |

## Why it did not transfer (reading, not tested)

- T8's base walked terrain (IMU in the obs, kp 40 / 32 Nm, a terrain
  curriculum) and failed by falling; the fine-tune taught it to fall less.
  This base is blind, flat-trained and capped at 12 Nm: it does not fall,
  it stops. Fine-tuning on hard levels at lr 1e-4 moves it further toward
  standing, which is what the hard levels reward under its upstream reward
  (stand-still and energy terms, no travel term beyond command tracking).

## Proposals

1. Robot team: check the inferred contract keys against the RPi's
   `policy_meta.json`; until then the robot profile's contract is a
   reconstruction.
2. Fine-tune on the levels where the base still moves: `--generator easy`
   with grow mutations from near-flat, or cap the level difficulty (e.g.
   d ≤ 0.5), so ACCEL's buffer stays inside this policy's reach; judge on
   flat + d0.25-d0.5 first.
3. A travel-aware score or reward for the fine-tune (learnability of
   success already needs travel; the PPO reward does not), or a lower lr
   (3e-5) to limit drift from the base.
4. If terrain capability matters on this robot, train the base with
   terrain in upstream itself (its tile curriculum on this contract)
   before any UED fine-tune.

## Cost

7.5 GPU-h (sacct, lem-gpu-short H100): kp 20 base (unused, submitted
before the target changed) 1.22, robot base 1.22, training 6 × 0.48 =
2.85, evals 7 × 0.32 = 2.23; CPU jobs (setup, battery) minutes.

## Files

`experiment.toml` (criterion, jobs), `record.md`, `upstream_base.sbatch`,
`setup.sbatch`, `train.sbatch`, `eval.sbatch`, `battery.sbatch`,
`to_upstream_run.py`, `analyze.py` → `results.json`. Eval outputs:
`$PDW/runs/robot_ft/`; checkpoints `$PDW/artifacts/robot_ft/`; base
`$PDW/copies/rft-0895699/artifacts/upstream/robot_base.pkl`.
