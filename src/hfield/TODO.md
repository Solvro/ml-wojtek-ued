# hfield / ACCEL — TODO

- **Richer terrain, not just hfield.** Goal is a walking policy that
  transfers to real terrain, incl. stairs and gaps — a pure heightfield
  can't represent those (discrete risers, holes). Current hfield-only
  genome (mountains/valleys) is step one. Next: extend the level
  representation to include discrete features (boxes for stairs, gaps),
  which likely means editing more than `hfield_data` per level and
  probably a rebuild step for structural changes, not just a runtime
  swap (see hfield_size being static pytree metadata).

- **Per-env level (vectorized ACCEL).** Today one level is shared by the
  whole `envs_num` batch per iteration (`hfield_data` in_axes=`None`).
  Give each env its own level instead: `in_axes=0`, batch of
  `(envs_num,)` levels/regrets through `generate_branch`/`replay_branch`
  and the `Curator`. More independent regret signal per iteration →
  faster convergence. Needs `Curator`/branch functions reworked from
  scalar level+score to vectorized batches.
