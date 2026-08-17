# PAIRED (learned adversary)

An alternative UED curriculum to ACCEL: instead of a random generator +
regret-curated replay buffer, a **learned adversary** designs the heightfield
levels, trained to maximise the regret between two students. Implements PAIRED
(Dennis et al., 2020) plus the two robustness fixes from *Stabilizing
Unsupervised Environment Design with a Learned Adversary* (see
[../stabilizing_ued_with_learned_adversary.md](../stabilizing_ued_with_learned_adversary.md)).

Shares the env stack, `A2C` student, PPO helpers, heightfield genome, and
`PlateauStopper` with ACCEL ([ued-accel.md](ued-accel.md)); the pieces below
are what's new.

## Three agents

| Agent | Module | Role |
|---|---|---|
| **Protagonist** | `A2C` (`modules/ppo_modules.py`) | the deployable student; **this is what gets checkpointed** |
| **Antagonist** | `A2C` | a second student, rolled out on the same levels; its best return estimates what the level *can* yield |
| **Generator** | `LevelGenerator` (`modules/generator.py`) | the learned adversary that designs levels |

Each has its own optimizer / `opt_state`. Only the protagonist is saved (via
the existing `meta.json` convention, so `scripts/eval.py` /
`scripts/eval_levels.py` restore it unchanged — `--algo paired`).

## The generator — `modules/generator.py`

A **single-step, continuous environment designer**. It maps a latent seed
`z ~ N(0, I)` (`latent_dim`) through an MLP to a diagonal Gaussian over the
`coarse_n²` genome cells, squashed by a `Sigmoid` bijector into `[0, 1]` — the
same Transformed-distribution pattern `A2C` uses for Tanh-squashed actions. The
whole level is emitted in one shot, so the teacher MDP has **horizon 1** (a
contextual bandit). This deliberately sidesteps the original PAIRED teacher's
hardest problem — sparse reward + credit assignment over a long level-building
episode (Section 3 of the paper). `entropy` is the `-log_prob` proxy (the
squashed distribution has no analytic entropy), matching `A2C`.

- `__call__(z, key)` → `(theta, log_prob, entropy)` — sample a level.
- `evaluate(z, theta)` → `(log_prob, entropy)` — score a stored `theta` for the
  PPO update.

## Regret — `ued/paired.py:paired_regret`

Per generated level, over its `envs_per_lane` episodes:

```
regret = max(antagonist_max_return − protagonist_mean_return, 0)
```

(matching the DCD reference). This is the generator's **reward** — it is
trained to maximise it. It differs from ACCEL's `positive_value_loss` regret
(a single-agent PLR proxy): PAIRED regret is a *two-agent* quantity, the gap
between what the antagonist achieves and what the protagonist averages.

## One iteration — `ued/paired.py:paired_iteration`

No generate/replay branches and no level buffer (unlike ACCEL) — pure PAIRED
regenerates every iteration:

1. `sample_levels`: generator turns `num_lanes` latent seeds into `num_lanes`
   levels; convert to `hfield_data` (numpy, per level).
2. `rollout` (`lanes_rollout`) both students on the **same** levels
   (`num_lanes × envs_per_lane` envs each).
3. `paired_regret` → per-level regret = generator reward; `prot_mean` feeds the
   stopper's reward signal.
4. **Students** — flatten lanes, GAE, `run_student_epochs` (PPO) for each. Both
   get the HiEnt entropy bonus (`student_entropy_coef`).
5. **Generator** — `run_generator_epochs`: single-step PPO on
   `(z, theta, old_log_prob, regret)` with a batch-mean-regret baseline (no GAE,
   no minibatching) and the HiEnt bonus (`gen_entropy_coef`).

Rollouts are off-tape exactly as in ACCEL ([ued-accel.md](ued-accel.md#stop-gradient));
the generator learns *only* through the regret reward via its single-step
policy gradient, never by differentiating through the environment.

## The Stabilizing-UED modifications

The paper proposes three modifications for its two problems; all three are
implemented here:

| Paper section | Targets | Where |
|---|---|---|
| 4.1 Entropy bonus (HiEnt) | P1 (entropy collapse / limited exploration) | `gen_entropy_coef` + `student_entropy_coef` below |
| 4.2 Optimizer (Evo) | P1 (blackbox exploration) | separate variant, `scripts/paired_evo.py` (below) |
| 4.3 Behavioural cloning (UniBC/BiBC) | P2 ("fall behind") | `bc_mode` below |

### HiEnt — high entropy bonus

Fights the teacher's entropy collapse (Problem P1). `gen_entropy_coef`
(default `0.1`, high) on the generator, plus `student_entropy_coef` on both
students. Both are `-log_prob` proxies added to the respective PPO losses.

### UniBC / BiBC — protagonist-antagonist behavioural cloning

Online policy distillation (paper Section 4.3), so the protagonist can learn
from the antagonist's solution instead of exploring from scratch (Problem P2,
"fall behind"). Adds a `KL(π_teacher || π_student)` term to the student PPO loss:

- `bc_mode="uni"` — protagonist distils from the antagonist only.
- `bc_mode="bi"` — additionally the antagonist distils from the (just-updated)
  protagonist.
- `bc_mode="none"` — vanilla PAIRED (KL still computed but weighted `0`).

Because both policies squash the *same* base Gaussian through the *same*
deterministic Tanh bijector, `KL` of the squashed action distributions equals
`KL` of the base `MultivariateNormalDiag`s (`ued/paired.py:_policy_kl`); the
teacher is `stop_gradient`-ed, so only the student receives gradient. The term
is applied every `kl_update_step` iterations (periodic distillation) via a
`bc_coef` scalar threaded into the update (0 on off-iterations — no recompile,
no branch). The paper's best results are **UniBC/BiBC + HiEnt together**.

## Trainer — `scripts/paired.py`

Same optax/jit/`FunctionalEnv` scaffolding as `scripts/accel.py` (accel arena,
`hfield_data`-swapping env, `movement_overrides()`/`terrain_overrides()`), plus:
two students + a generator with three optimizers, a `PairedTrainer` (jitted
callables + hyperparameters), and a `PlateauStopper` fed `regret_score` +
`mean_reward` (the protagonist's). Train env is sized to
`num_lanes × envs_per_lane`; eval runs the protagonist on the default terrain.
`ALGO = "paired"` → `artifacts/paired/<run>/`. Logs `regret_score`,
`max_regret_score`, `prot_return`, `antag_return`, `prot_loss`, `antag_loss`,
`gen_loss`, `bc_coef`, `smoothed_regret`/`smoothed_reward`.

Run with `just paired` (needs `just build-accel-arena` first). All knobs
(`PairedParams` + reused PPO params): [configuration.md](configuration.md).

## PAIRED+Evo (Section 4.2) — `scripts/paired_evo.py`

The paper's Section 4.2 argues *"RL may not be the best fit for this blackbox
optimization problem"* and replaces the learned adversary with a **search-based
editor**: ACCEL's random generate + mutate + regret-curated buffer, but scored
by the **PAIRED two-agent regret** instead of ACCEL's single-agent
`positive_value_loss`. So this variant has **no `LevelGenerator`** — the level
design is pure search, and the only learned networks are the two students.

It reuses the ACCEL curriculum machinery wholesale (`ued.curriculum.decide_replay`,
`utils.buffer.LevelBuffer`, `ued.types.Curator`, `HFieldManager.generate/mutate`)
and layers the PAIRED regret + antagonist + student BC on top
(`ued/paired.py`: `PairedEvoTrainer`, `paired_evo_generate_branch`,
`paired_evo_replay_branch`, `score_regret_evo`). Same generate/replay split as
vanilla ACCEL ([ued-accel.md](ued-accel.md#the-two-branches-algorithm-1-vanilla-scriptsaccelpy)):

- **generate** (no policy update): a random level; roll out *both* students;
  regret `= max(antag_max − prot_mean, 0)`; curate into the buffer by regret.
- **replay** (trains both students): sample a stored level; roll out and
  PPO-update both students (HiEnt + optional BC); mutate the level; re-score the
  child's PAIRED regret and curate it.

Regret is single-level here (`paired_regret_single`, over `envs_num` episodes),
not per-lane. Everything else — HiEnt (`student_entropy_coef`), UniBC/BiBC
(`bc_mode`), the protagonist-as-deliverable checkpoint, and the dual-signal
`PlateauStopper` — is shared with the learned-adversary script. `ALGO =
"paired_evo"` → `artifacts/paired_evo/<run>/`.

Run with `just paired-evo`. Knobs: `PairedEvoParams`
([configuration.md](configuration.md#paired_evo)).

## Untuned defaults

The `PairedParams` magnitudes (`latent_dim`, `gen_learning_rate`,
`gen_entropy_coef`, `bc_coef`, `kl_update_step`, `gen_epochs`) are reasoned
starting points, not validated on this robot — treat them as a baseline to
tune, like the `env_overrides` magnitudes ([ppo.md](ppo.md#shared-env-overrides--utilsenv_overridespy)).
