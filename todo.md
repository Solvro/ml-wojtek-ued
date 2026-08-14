# Todo

- Evolutionary mutation step for ACCEL: per replay iteration, generate `M`
  mutations of the same parent level `θ` instead of one, keep only the
  highest-regret mutant. Population-based edit/selection instead of a single
  random-walk mutation. Pairs with the batched-lanes/top-k curation
  (`src/ued/curriculum.py`). Not started.
