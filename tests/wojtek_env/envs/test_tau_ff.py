"""Feed-forward torque head: the model-free pieces.

The env-side plumbing (24-wide actions, qfrc_applied under both delay
paths) needs a live env; here we pin the swing-cost and target-sag math
and the config defaults.
"""

import jax.numpy as jp
import numpy as np

from wojtek_env.envs import env as joystick


def test_swing_cost_zero_when_all_feet_down():
    tau = jp.ones(12) * 2.0
    contact = jp.ones(4, dtype=bool)
    assert float(joystick.tau_ff_swing_cost(tau, contact)) == 0.0


def test_swing_cost_full_when_airborne():
    tau = jp.arange(12.0) - 6.0  # mixed signs: the cost is |tau|
    contact = jp.zeros(4, dtype=bool)
    expected = float(jp.sum(jp.abs(tau)))
    np.testing.assert_allclose(
        float(joystick.tau_ff_swing_cost(tau, contact)), expected
    )


def test_swing_cost_gates_per_leg():
    # 1 Nm on every joint; legs 0 and 2 airborne -> 2 legs * 3 joints.
    tau = jp.ones(12)
    contact = jp.array([False, True, False, True])
    np.testing.assert_allclose(float(joystick.tau_ff_swing_cost(tau, contact)), 6.0)
    # The cost must charge the AIRBORNE legs' torque, not the stance legs'.
    tau_stance_only = jp.zeros(12).at[3:6].set(5.0).at[9:12].set(5.0)
    np.testing.assert_allclose(
        float(joystick.tau_ff_swing_cost(tau_stance_only, contact)), 0.0
    )


def test_target_sag_gates_on_stance():
    # 0.1 rad of sag on every joint; legs 1 and 3 in contact -> only their
    # 2*3 joints are charged.
    q = jp.zeros(12)
    targets = jp.full(12, 0.1)
    contact = jp.array([False, True, False, True])
    np.testing.assert_allclose(
        float(joystick.target_sag_cost(q, targets, contact)),
        6 * 0.1**2,
        rtol=1e-6,
    )
    # Swing legs are free: sag there costs nothing.
    q_swing_only = jp.zeros(12).at[0:3].set(-0.3).at[6:9].set(0.3)
    np.testing.assert_allclose(
        float(joystick.target_sag_cost(q_swing_only, jp.zeros(12), contact)),
        0.0,
    )


def test_target_sag_zero_when_tracking():
    q = jp.arange(12.0) * 0.1
    contact = jp.ones(4, dtype=bool)
    assert float(joystick.target_sag_cost(q, q, contact)) == 0.0


def test_tau_ff_disabled_by_default():
    cfg = joystick.default_config()
    assert not cfg.tau_ff.enable
    assert cfg.reward.scales.tau_ff_swing == 0.0
    assert cfg.reward.scales.tau_ff == 0.0
