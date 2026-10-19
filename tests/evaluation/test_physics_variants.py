from pathlib import Path

import mujoco
import numpy as np
import pytest

from evaluation import physics_variants as pv


def test_mjx_match_is_an_independent_copy(model: mujoco.MjModel) -> None:
    m = pv.mjx_match(model, None)
    m.opt.timestep = 1.0
    assert model.opt.timestep == pytest.approx(0.004)


def test_newton_elliptic_sets_converged_solver(model: mujoco.MjModel) -> None:
    m = pv.newton_elliptic(model, None)
    assert m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert (m.opt.iterations, m.opt.ls_iterations) == (100, 50)
    assert model.opt.iterations == 2


def test_half_dt_keeps_effective_solref(model: mujoco.MjModel) -> None:
    m = pv.half_dt(model, None)
    assert m.opt.timestep == pytest.approx(0.002)
    # 0.001 s ran at the 2 x 0.004 s floor in training: pinned there
    np.testing.assert_allclose(m.geom_solref[model.geom("box").id], [0.008, 2])
    np.testing.assert_allclose(m.eq_solref[0], [0.008, 1])
    leg = model.geom("leg_collision").id
    np.testing.assert_allclose(m.geom_solref[leg], [0.02, 1])
    # direct (negative) stiffness/damping is not a time constant
    np.testing.assert_allclose(
        m.geom_solref[model.geom("link_geom").id], [-1000, -10]
    )


def test_soft_contact_scales_run_timeconst(model: mujoco.MjModel) -> None:
    m = pv.soft_contact(model, None)
    scale = pv.SOFT_TIMECONST_SCALE
    # 0.001 s runs at the 2 x 0.004 s floor: scaled from there
    np.testing.assert_allclose(
        m.geom_solref[model.geom("box").id], [0.008 * scale, 2]
    )
    leg = model.geom("leg_collision").id
    np.testing.assert_allclose(m.geom_solref[leg], [0.02 * scale, 1])
    link = model.geom("link_geom").id
    np.testing.assert_allclose(m.geom_solref[link], [-1000, -10])
    np.testing.assert_allclose(m.geom_solimp, model.geom_solimp)
    np.testing.assert_allclose(m.eq_solref, model.eq_solref)


def test_soft_contact_without_refsafe_scales_raw_timeconst(
    model: mujoco.MjModel,
) -> None:
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_REFSAFE
    m = pv.soft_contact(model, None)
    box = model.geom("box").id
    np.testing.assert_allclose(
        m.geom_solref[box], [0.001 * pv.SOFT_TIMECONST_SCALE, 2]
    )


def test_full_collision_meshes_hit_terrain_only(
    tmp_path: Path, robot_xml: str
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(robot_xml)
    m = pv.wojtek_full_collision_spec(scene).compile()
    leg, box = m.geom("leg_collision").id, m.geom("box").id
    terrain, marker = m.geom("terrain").id, m.geom("marker").id

    def collide(a: int, b: int) -> bool:
        return bool(
            m.geom_contype[a] & m.geom_conaffinity[b]
            or m.geom_contype[b] & m.geom_conaffinity[a]
        )

    assert collide(leg, terrain)
    assert not collide(leg, box)
    assert not collide(leg, marker)
    assert m.mesh_graphadr[m.geom_dataid[leg]] >= 0


def test_copy_tuning_restores_constructor_edits(model: mujoco.MjModel) -> None:
    trained = pv.mjx_match(model, None)
    trained.actuator_gainprm[:, 0] = 7.0
    trained.opt.timestep = 0.002
    fresh = pv.copy_tuning(trained, pv.mjx_match(model, None))
    assert fresh.actuator_gainprm[0, 0] == 7.0
    assert fresh.opt.timestep == pytest.approx(0.002)
