# deploy-ab: why deploy-profile training collapses, and what to run

- **Date:** 2026-10-15. **Code:** `jgilewicz/deploy-ab` (frozen copy
  `deploy-ab-35b0594`), upstream w01-tek `e44c1a9` (frozen copy
  `w01tek-e44c1a9`). **wandb:** `wojtyla`, tag `deploy-ab`.
- **Cost:** 12.1 GPU-h (arms 8.9, final evals 2.1, upstream 1.1 incl. a
  1.7-min OOM). Jobs: build 6108905, arms 6108906_[0-4], evals
  6108908_[0-4], A1 6108929 (6108907 OOM at 32768 envs), A1-long 6109504.
- Every arm: seed 0, deploy profile, inference 1:1 upstream
  (`tests/scripts/test_env_override.py` checks every arm file against the
  snapshot's CONSUMED keys and network).

## Verdict

- **The collapse is the upstream recipe's own early phase, not our loop.**
  Upstream's `train.py` with `+experiment=terrain_blind_v4_4d`, from
  scratch (A1/A1-long), drops to ~10-step episodes ending base-down by
  11M steps, stays there to ~40M, then recovers: eval episode length 166
  at 51M, 478 at 102M, 754 at 152M, ~900 from 190M to 305M (eval reward
  -777 → +58, base-down ends 0.8 % → 0).
- **Our loop is in the same state at 50M.** A0/A3/A4/A5 at 50M: ~0.08-0.1
  falls/step (≈10-step episodes), eval fall rate ≈1.0. A1-long at 50.8M:
  train episode length 9.3 (≈0.11 falls/step), 75 % of eval episodes end
  base-down. T4 was cancelled at 32M, inside this phase.
- **Mechanism (A2, A6, diagnostic only):** the jitter penalty of the
  stochastic initial policy (`torque_rate` -0.02 ≈ -1.15 to -1.4/step)
  dwarfs the one-off termination (-1 x 0.02 = -0.02), so dying early is
  the best response until the policy is smooth enough. With torque_rate
  at v10's -1e-4 (A6) or the whole v10 reward (A2) the dip is shorter
  and shallower and both are recovering by 50M (A2: 0.001 falls/step,
  eval d0.25 fall 0; A6: 0.013 falls/step, eval d0.25 fall 0.34). Not
  candidate fixes: they change the smoothness/gait shaping that works on
  hardware.
- **No upstream-matching arm shortens the dip within 50M:** A3 (1000-step
  levels), A4 (clip 0.3), A5 (flat levels) are all still collapsed at 50M,
  as is A1. Healthy by the brief's bar (falls/step < 0.02 and eval d0.25
  fall < 0.3 at 50M): none of the candidate arms; A2 only (diagnostic).
- **A7 not run:** its trigger (A1 healthy, A3/A4/A5 not) did not hold —
  A1 itself is collapsed at 50M.

## Recommended config for the 3-seed deploy run

- `brax_ued.py --profile deploy --mode accel --level_space param --n_edits 2
  --edit_scale 0.1 --clipping_epsilon 0.3 --num_timesteps 300000000`, no
  `--env_override`: upstream's reward weights and DR untouched, PPO clip
  matched to upstream's brax default.
- Budget for the recovery: upstream needs ~150M steps to episode length
  > 750 and ~190M to plateau. Our loop runs ~9k steps/s on one H100 →
  300M ≈ 9.3 h/seed, ~28 GPU-h for 3 seeds + ~1.3 GPU-h evals.
- Go/no-go at ~150M (≈4.7 h): train falls/step should be falling towards
  < 0.02 and eval d0.25 fall rate below ~0.5, as in A1-long; if still at
  ~0.1 falls/step, stop and escalate (brax_ued saves only at the end).
- Alternative if that is too slow: train with upstream's trainer itself
  (A1-long reached a walking policy in 55 min / 305M steps), then evaluate
  and export through our pipeline — needs a converter from upstream's
  checkpoint to our sidecar/`.pkl` (not built).

## Metrics

Train metrics per iteration (2.46M steps; A3 8.2M), in-loop eval on 2x512
fresh levels, 300-step levels (A3: 1000-step, so its in-loop eval is not
comparable; use its final eval). Top-3 terms = env `reward/<term>` x scale
x dt, per step. A0 and v10 predate `reward_term/*` logging.

### 10M env steps

| arm | falls/step | train success | reward/step | steps/s | d0.25 success | d0.25 fall | d0.25 travel | d1.0 success | d1.0 fall | d1.0 travel | top-3 terms / step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v10 seed4 (benchmark) | 0.0264 | 0 | 0.0203 | 7.1k | - | - | - | - | - | - | - |
| A0 deploy (T4 seed0) | 0.00156 | 0.101 | -0.845 | 7.0k | - | - | - | - | - | - | - |
| A2 v10 reward | 0.0411 | 0 | -0.0296 | 6.3k | 0 | 0.732 | 0.031 | 0 | 0.72 | 0.032 | action_accel -0.014, action_rate -0.013, height_tracking +0.012 |
| A3 level_steps 1000 | 0.00153 | 0 | -1.42 | 9.3k | 0 | 0.108 | 0.10 | 0 | 0.407 | 0.316 | torque_rate -1.164, torque_limit -0.123, stand_still -0.089 |
| A4 clip 0.3 | 0.0266 | 0.002 | -0.596 | 6.3k | 0 | 0.633 | 0.037 | 0 | 0.617 | 0.039 | torque_rate -0.481, stand_still -0.062, torque_limit -0.052 |
| A5 flat levels | 0.0491 | 0 | -0.608 | 6.8k | 0 | 1 | 0.032 | 0 | 1 | 0.032 | torque_rate -0.485, stand_still -0.064, torque_limit -0.051 |
| A6 torque_rate -1e-4 | 0.00135 | 0.040 | -0.13 | 6.3k | 0 | 0.038 | 0.065 | 0 | 0.041 | 0.066 | torque_limit -0.067, stand_still -0.057, tracking_lin_vel +0.024 |

### 25M env steps

| arm | falls/step | train success | reward/step | steps/s | d0.25 success | d0.25 fall | d0.25 travel | d1.0 success | d1.0 fall | d1.0 travel | top-3 terms / step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v10 seed4 (benchmark) | 0.00675 | 0.777 | 0.0359 | 9.6k | 0.786 | 0.089 | 1.58 | 0.196 | 0.438 | 0.725 | - |
| A0 deploy (T4 seed0) | 0.0963 | 0 | -0.326 | 9.6k | 0 | 1 | 0.012 | 0 | 1 | 0.011 | - |
| A2 v10 reward | 0.00656 | 0 | 0.0174 | 8.1k | 0 | 0.121 | 0.072 | 0 | 0.112 | 0.071 | height_tracking +0.018, feet_phase +0.013, contact_match +0.010 |
| A3 level_steps 1000 | 0.0845 | 0 | -0.382 | 7.8k | 0 | 1 | 0.026 | 0 | 1 | 0.026 | torque_rate -0.293, stand_still -0.063, torque_limit -0.028 |
| A4 clip 0.3 | 0.105 | 0 | -0.191 | 8.1k | 0 | 1 | 0.010 | 0 | 1 | 0.009 | torque_rate -0.135, stand_still -0.062, tracking_lin_vel +0.019 |
| A5 flat levels | 0.0968 | 0 | -0.151 | 8.4k | 0 | 1 | 0.027 | 0 | 1 | 0.027 | torque_rate -0.111, stand_still -0.052, tracking_lin_vel +0.027 |
| A6 torque_rate -1e-4 | 0.0611 | 0.004 | -0.0314 | 8.1k | 0.008 | 0.789 | 0.078 | 0.002 | 0.834 | 0.059 | stand_still -0.039, tracking_lin_vel +0.027, torque_limit -0.020 |

### 50M env steps (A0 was cancelled at 32M)

| arm | falls/step | train success | reward/step | steps/s | d0.25 success | d0.25 fall | d0.25 travel | d1.0 success | d1.0 fall | d1.0 travel | top-3 terms / step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v10 seed4 (benchmark) | 0.000416 | 0.841 | 0.0487 | 10.6k | 0.851 | 0.029 | 1.84 | 0.247 | 0.372 | 0.802 | - |
| A2 v10 reward | 0.00103 | 0.157 | 0.0392 | 9.0k | 0.163 | 0 | 0.352 | 0.033 | 0.054 | 0.218 | height_tracking +0.019, feet_phase +0.013, tracking_lin_vel +0.010 |
| A3 level_steps 1000 | 0.0949 | 0 | -0.226 | 8.1k | 0 | 1 | 0.025 | 0 | 1 | 0.025 | torque_rate -0.162, stand_still -0.060, tracking_lin_vel +0.018 |
| A4 clip 0.3 | 0.104 | 0 | -0.103 | 9.0k | 0 | 1 | 0.014 | 0 | 1 | 0.015 | torque_rate -0.085, stand_still -0.046, tracking_lin_vel +0.025 |
| A5 flat levels | 0.0786 | 0 | -0.0896 | 9.1k | 0 | 0.996 | 0.037 | 0 | 0.995 | 0.033 | torque_rate -0.081, stand_still -0.043, tracking_lin_vel +0.035 |
| A6 torque_rate -1e-4 | 0.013 | 0.291 | 0.0407 | 9.0k | 0.345 | 0.34 | 0.762 | 0.11 | 0.407 | 0.404 | tracking_lin_vel +0.043, tracking_ang_vel +0.026, stand_still -0.016 |

### Final eval (`eval_brax.py`, 50M checkpoint, 300-step levels, 512 levels)

`cmd_success_rate` = no fall and the sampled command tracked (PR #23).

| arm | d0.25 success | d0.25 cmd_success | d0.25 fall | d1.0 success | d1.0 cmd_success | d1.0 fall | flat success | flat cmd_success |
|---|---|---|---|---|---|---|---|---|
| v10 seed4 (300M, reference) | 0.909 | - | 0.016 | 0.423 | - | 0.358 | 0.936 | - |
| A2 v10 reward | 0.169 | 0.343 | 0.000 | 0.021 | 0.221 | 0.042 | 0.148 | 0.320 |
| A3 level_steps 1000 | 0 | 0 | 1.000 | 0 | 0 | 1.000 | - | - |
| A4 clip 0.3 | 0 | 0 | 1.000 | 0 | 0 | 1.000 | 0 | 0 |
| A5 flat levels | 0 | 0 | 0.998 | 0 | 0 | 1.000 | 0 | 0 |
| A6 torque_rate -1e-4 | 0.346 | 0.367 | 0.323 | 0.115 | 0.192 | 0.400 | 0.336 | 0.359 |

### A1 / A1-long: upstream trainer (brax `ppo.train`, warp, 8192 envs)

Eval = brax evaluator (stochastic policy, brax default), 128 envs x
1000-step episodes. Upstream's
termination term is zeroed inside its 1 s spawn grace, so falls are read
from `base_contact_at_done` (episode ended base-down).

| env steps | eval ep_len | eval ends base-down | eval no-progress cut | eval reward | train ep_len | steps/s | top-3 terms / step |
|---|---|---|---|---|---|---|---|
| 0 | 555 | 0.125 | 0.68 | -777 | 198 | - | torque_rate -1.149, torque_limit -0.119, stand_still -0.103 |
| 10.2M | 111 | 0.82 | 0.18 | -14 | 9.5 | 59k | torque_rate -0.124, tracking_lin_vel +0.025, stand_still -0.025 |
| 20.3M | 44 | 0.945 | 0.047 | -0.83 | 23.3 | 75k | torque_rate -0.044, tracking_lin_vel +0.033, stand_still -0.029 |
| 30.5M | 57 | 0.906 | 0.094 | -0.82 | 8.3 | 89k | torque_rate -0.039, tracking_lin_vel +0.032, stand_still -0.022 |
| 50.8M | 166 | 0.75 | 0.188 | 2.6 | 9.3 | 89k | tracking_lin_vel +0.035, torque_rate -0.022, tracking_ang_vel +0.021 |
| 101.6M | 478 | 0.203 | 0.664 | 13.0 | 110 | 91k | tracking_lin_vel +0.033, tracking_ang_vel +0.020, torque_rate -0.019 |
| 152.4M | 754 | 0.047 | 0.461 | 42.3 | 595 | 99k | tracking_lin_vel +0.045, tracking_ang_vel +0.027, torque_rate -0.019 |
| 193.0M | 905 | 0.008 | 0.219 | 57.9 | 892 | 107k | tracking_lin_vel +0.052, tracking_ang_vel +0.031, torque_rate -0.022 |
| 304.7M | 907 | 0.008 | 0.188 | 58.0 | 840 | 117k | tracking_lin_vel +0.057, tracking_ang_vel +0.034, torque_rate -0.029 |

A1 (50M, separate run, same config): eval ep_len 521 → 26 (11M) → ~10
(17-39M) → 26 (50M), 98-100 % base-down ends from 11M.

## Our loop vs upstream's trainer: remaining differences

| # | setting | upstream `train.py` | brax_ued deploy | matched? |
|---|---|---|---|---|
| 1 | env / reward / DR config | `terrain_blind_v4_4d` | same (snapshot test) | yes, except terrain + sim below |
| 2 | network, init, obs normalization | Playground Go1 factory, SiLU, tanh-normal | same kwargs | yes |
| 3 | PPO: unroll 20, 32 minibatches, 4 epochs, lr 3e-4, entropy 1e-2, discount 0.97, GAE 0.95, grad norm 1.0, reward_scaling 1, normalize_advantage | Playground Go1 | same | yes |
| 4 | PPO clip | 0.3 (brax default) | 0.2 | flag `--clipping_epsilon 0.3` (A4); recommended |
| 5 | envs per update | 8192 (A1; train.sh's 32768 is a multi-GPU total, OOM on 1 H100) | 256 lanes x 32 = 8192 | yes for A1; upstream's own runs' scale is not recorded |
| 6 | episode | 1000 steps, async auto-reset, envs persist across updates (`num_resets_per_eval=0`) | `level_steps` 300, all envs reset onto new levels each iteration | A3 matches the length; the async/persistent part is not |
| 7 | terrain curriculum | tile arena, flat row, `pinned_flat_frac` 0.1, feature spawns, promotion/demotion | ACCEL param levels (A0: genome dist 0.035 = near-flat), 12x12 m arena | not matched; A5 = all-flat bound |
| 8 | spawn grace | termination penalty zeroed for 1 s after spawn | none (terrain block replaced) | no; ours prices early falls slightly more |
| 9 | command terrain bias | terrain rows lean forward | same config, our arena's flat/terrain split differs | partly |
| 10 | DR draws | once per env for the whole run | redrawn every iteration | no |
| 11 | ACCEL explore iterations | n/a (every step trains) | ~20 % of iterations score levels without a PPO update | no (A5 trains every iteration) |
| 12 | physics backend | warp | jax (per-env hfield swap; warp collides every env with env 0's level) | no, cannot be matched with per-env levels |
| 13 | warm start | none (`restore=null`; family trains from scratch) | none | yes |
| 14 | step budget | preset 2B steps | 300M planned | — |

## Throughput

- Upstream trainer: 89-117k env steps/s on one H100 (A1-long: 305M in
  54 min). brax_ued: 8-10k steps/s (~9x slower).
- Not ACCEL: A5 (flat levels, `dr_easy`, every iteration trains) runs at
  the same 8-9k as the ACCEL arms. Likely causes, not profiled: the jax
  MJX backend with heightfield collision vs warp, and the per-env model
  swap (`hfield_data` + DR model vmapped per env) in `FunctionalEnv`;
  plus 3 arms sharing a 4-GPU node.
- Consequence: 2B steps (upstream's preset budget) would take ~62 h/seed
  in our loop; 300M (≈150M past the recovery point) takes ~9.3 h/seed.

## Caveats

- One seed per arm; A1-long's recovery point (~45M start, ~150M healthy)
  is one run. Our loop's recovery was not observed (all candidate arms
  stopped at 50M) — the recommendation's go/no-go check covers that.
- A2's v10 reward includes `feet_phase`/`contact_match`, which the deploy
  actor cannot observe (no `phase` input); it still recovered first.

- Report page: https://claude.ai/code/artifact/b7670145-1c80-4829-916f-d42c3b3c03aa
