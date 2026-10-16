# robot-ft — record

`PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued`.
Upstream: `/home/jakgil6519/workspace/w01tek-e44c1a9` (w01-tek `e44c1a9`).

## Target contract

- Brief: the policy pinned in `policy_source.py` (`wojtek-quiet-locomotion@553795b`
  = `wojtek_flat_quiet_v10_s0`, w01-tek c843245: kp 20 / kd 0.5 / 12 Nm,
  tau_ff, no IMU, no action filter). No `flat_quiet_v10` preset exists on any
  branch; `flat_quiet_v6` has that contract.
- 2026-10-16 ~15:25, coordinator (user decision): target the policy the RPi
  actually loads instead — the drop-in `ros/deploy/rpi/wojtek-robot-local.conf`
  launches `policy:=/home/rpi/policy` = `wojtek_flat_quiet_baseacc_s0`.

- Coordinator answer (msg_7079ee6260c0): proposal approved; the RPi meta
  cannot be fetched. Robot profile = resolved `flat_quiet_v6` + `pd_kp=40`,
  `pd_kd=0.8`; base = upstream `train.py` with those plus the `base_accel`
  dose (`brax_ued.UPSTREAM_RUNS["robot"]`), from scratch, 2B, seed 0.
  Budget ~16 GPU-h including 6113986.

| date | job | what |
|---|---|---|
| 2026-10-16 15:28 | 6113986 | upstream `train.py +experiment=flat_quiet_v6` (kp 20), 2B, warp, 8192 envs; submitted before the HOLD |
| 2026-10-16 ~16:50 | 6113986 | (finished 16:40, 1.22 GPU-h; wrong plant, not used) |
| 2026-10-16 ~16:55 | 6114800 | robot base: upstream `train.py` `UPSTREAM_RUNS["robot"]`, 2B, run `rft_upstream_seed0` |
