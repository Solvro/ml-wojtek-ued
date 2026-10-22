# Documentation

Quadruped UED Benchmark: Unsupervised Environment Design for quadruped
locomotion on procedural terrain. Start with the [README](../README.md)
(quickstart, reproduce the paper), then:

| Page | Covers | Code |
|---|---|---|
| [overview.md](overview.md) | methods, robots, level space, metrics, pipeline, repo map, glossary | — |
| [setup.md](setup.md) | requirements, pinned dependencies, install, generated assets, outputs | `justfile` |
| [environment.md](environment.md) | `wojtek_env`, task knobs, `FunctionalEnv`, tiled arena, warp/jax backends | `src/wojtek_env/`, `src/env/` |
| [levels.md](levels.md) | level genome, `HFieldConfig`, arena assets, level spaces | `src/hfield/` |
| [training.md](training.md) | trainer, the 12 curricula, PAIRED, throughput | `src/training/` |
| [robots.md](robots.md) | Go1, Go2, Barkour vB, Spot on the tiled arena | `src/robots/` |
| [evaluation.md](evaluation.md) | final eval, sim2sim, demo videos | `src/evaluation/`, `src/rendering/` |
| [results.md](results.md) | paper tables, statistics (rliable), figures, domination comparison | `src/paper/`, `src/evaluation/domination.py` |
| [export.md](export.md) | train from one YAML, export to ONNX, the generated deployment README | `src/export/`, `conf/policy/` |
| [configuration.md](configuration.md) | every Hydra config group and field | `conf/`, `src/config/` |
| [slurm.md](slurm.md) | job variables, per-robot budgets, cluster values | `slurm/` |
| [development.md](development.md) | layout rules, recipes, tests, adding features, gotchas | — |

## By question

- Which methods and robots exist? [overview.md](overview.md#methods),
  [overview.md](overview.md#robots)
- How do I run one training run? [setup.md](setup.md#run),
  [configuration.md](configuration.md#trainyaml--trainconfig)
- How is a level represented and mutated? [levels.md](levels.md)
- Why can levels be swapped without recompiling?
  [environment.md](environment.md#why-the-arena-swap-is-free)
- Which backend for which robot? [slurm.md](slurm.md#per-robot-budgets),
  [robots.md](robots.md)
- What does each eval metric mean? [overview.md](overview.md#metrics),
  [evaluation.md](evaluation.md#final-eval)
- How do I get a policy onto the real robot?
  [export.md](export.md)
- How do I add a robot / curriculum / config field?
  [development.md](development.md#adding-features)
