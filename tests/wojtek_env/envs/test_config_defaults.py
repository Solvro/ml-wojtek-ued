"""Pins code defaults that presets override by key, so a renamed or
removed knob fails here instead of at env construction.
"""

from wojtek_env.envs import env as joystick


# The terrain gate has no yaml mirror (nothing in the reward block does), so
# this pins the code defaults themselves: a preset can only override keys
# that exist here.
def test_terrain_gate_defaults():
    assert joystick.default_config().reward.terrain_gate.to_dict() == {
        "enable": False,
        "floor": 0.15,
        "rough_ref": 0.05,
        "landing_soften": 0.5,
        "orientation_tol_flat_deg": 0.0,
        "flat_pitch_tol_deg": 2.0,
        "flat_pitch_rough_cut": 0.25,
        "flat_pitch_row_only": False,
        "flat_pitch_spin_exempt": False,
    }


# The v4.4 knobs are additive and must be inert by default: scale 0.0 makes
# flat_pitch a `+ raw*0.0` no-op, sticky False keeps the command stream
# bit-identical, and a 0.0 flat pin leaves the pinned mask unchanged.
def test_v44_knobs_default_off():
    cfg = joystick.default_config()
    assert cfg.reward.scales.flat_pitch == 0.0
    assert cfg.command.pure_wz_sticky is False
    assert cfg.terrain.pinned_flat_frac == 0.0
