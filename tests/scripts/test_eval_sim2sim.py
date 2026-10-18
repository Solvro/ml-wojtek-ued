from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np
import pytest

from hfield.level_space import make_level_space
from scripts import eval_sim2sim as s2s
from scripts.brax_ued import ROBOT_SCALE, hfield_config
from scripts.eval_brax import SUITE_STRENGTHS, SUITE_TERRAINS, suite_hfield

ROBOT_XML = """
<mujoco>
  <option timestep="0.004" iterations="2"/>
  <asset>
    <hfield name="hf" nrow="5" ncol="5" size="1 1 0.2 0.1"/>
    <mesh name="cube" vertex="0 0 0  .1 0 0  0 .1 0  0 0 .1  .1 .1 .1"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="hf" conaffinity="15"/>
    <geom name="marker" type="sphere" size=".01" pos="2 2 2"
      contype="0" conaffinity="0"/>
    <body name="base" pos="0 0 0.3">
      <freejoint/>
      <geom name="box" type="box" size=".1 .1 .05" solref="0.001 2"/>
      <body name="leg" pos="0.1 0 0">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="leg_collision" type="mesh" mesh="cube"
          contype="0" conaffinity="0" solref="0.02 1"/>
        <site name="foot"/>
      </body>
      <body name="link" pos="0.2 0 0">
        <joint name="link_hinge" axis="0 1 0"/>
        <geom name="link_geom" type="sphere" size=".02" solref="-1000 -10"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect body1="leg" body2="link" anchor="0.1 0 0" solref="0.001 1"/>
  </equality>
  <actuator><position joint="hinge" kp="20"/></actuator>
  <sensor>
    <framepos name="foot_pos" objtype="site" objname="foot"/>
    <gyro name="gyro" site="foot"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ROBOT_XML)


def test_mjx_match_is_an_independent_copy(model: mujoco.MjModel) -> None:
    m = s2s.mjx_match(model, None)
    m.opt.timestep = 1.0
    assert model.opt.timestep == pytest.approx(0.004)


def test_newton_elliptic_sets_converged_solver(model: mujoco.MjModel) -> None:
    m = s2s.newton_elliptic(model, None)
    assert m.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert (m.opt.iterations, m.opt.ls_iterations) == (100, 50)
    assert model.opt.iterations == 2


def test_half_dt_keeps_effective_solref(model: mujoco.MjModel) -> None:
    m = s2s.half_dt(model, None)
    assert m.opt.timestep == pytest.approx(0.002)
    # 0.001 s ran at the 2 x 0.004 s floor in training: pinned there
    np.testing.assert_allclose(m.geom_solref[model.geom("box").id], [0.008, 2])
    np.testing.assert_allclose(m.eq_solref[0], [0.008, 1])
    leg = model.geom("leg_collision").id
    np.testing.assert_allclose(m.geom_solref[leg], [0.02, 1])
    # direct (negative) stiffness/damping is not a time constant
    np.testing.assert_allclose(m.geom_solref[model.geom("link_geom").id], [-1000, -10])


def test_soft_contact_scales_run_timeconst(model: mujoco.MjModel) -> None:
    m = s2s.soft_contact(model, None)
    scale = s2s.SOFT_TIMECONST_SCALE
    # 0.001 s runs at the 2 x 0.004 s floor: scaled from there
    np.testing.assert_allclose(m.geom_solref[model.geom("box").id], [0.008 * scale, 2])
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
    m = s2s.soft_contact(model, None)
    box = model.geom("box").id
    np.testing.assert_allclose(
        m.geom_solref[box], [0.001 * s2s.SOFT_TIMECONST_SCALE, 2]
    )


def test_full_collision_meshes_hit_terrain_only(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(ROBOT_XML)
    m = s2s.wojtek_full_collision_spec(scene).compile()
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
    trained = s2s.mjx_match(model, None)
    trained.actuator_gainprm[:, 0] = 7.0
    trained.opt.timestep = 0.002
    fresh = s2s.copy_tuning(trained, s2s.mjx_match(model, None))
    assert fresh.actuator_gainprm[0, 0] == 7.0
    assert fresh.opt.timestep == pytest.approx(0.002)


def test_sensor_index_follows_names(model: mujoco.MjModel) -> None:
    spec = mujoco.MjSpec.from_string(ROBOT_XML)
    spec.add_sensor(
        name="extra",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="foot",
    )
    variant = spec.compile()
    np.testing.assert_array_equal(
        s2s.sensor_index(model, variant), np.arange(model.nsensordata)
    )
    reordered = mujoco.MjSpec.from_string(ROBOT_XML)
    reordered.delete(reordered.sensors[0])
    reordered.add_sensor(
        name="foot_pos",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="foot",
    )
    np.testing.assert_array_equal(
        s2s.sensor_index(model, reordered.compile()), [3, 4, 5, 0, 1, 2]
    )


def test_sensor_index_rejects_missing_sensor(model: mujoco.MjModel) -> None:
    spec = mujoco.MjSpec.from_string(ROBOT_XML)
    spec.delete(spec.sensors[1])
    with pytest.raises(ValueError, match="gyro"):
        s2s.sensor_index(model, spec.compile())


def test_cpu_physics_rejects_incompatible_models(model: mujoco.MjModel) -> None:
    physics = s2s.CpuPhysics(model, threads=1)
    other = mujoco.MjModel.from_xml_string(ROBOT_XML.replace("<freejoint/>", ""))
    with pytest.raises(ValueError, match="nq"):
        physics.use(other)
    odd = s2s.mjx_match(model, None)
    odd.opt.timestep = 0.003
    with pytest.raises(ValueError, match="divide"):
        physics.use(odd)


def specs_for(model: mujoco.MjModel) -> dict[str, jax.ShapeDtypeStruct]:
    data = mujoco.MjData(model)
    shapes = {f: np.shape(getattr(data, f)) for f in s2s.OUT_FIELDS}
    shapes["xmat"] = (model.nbody, 3, 3)
    shapes["site_xmat"] = (model.nsite, 3, 3)
    shapes["geom_xpos"] = (model.ngeom, 3)
    shapes["geom_xmat"] = (model.ngeom, 3, 3)
    return {f: jax.ShapeDtypeStruct(s, np.float32) for f, s in shapes.items()}


def host_inputs(model: mujoco.MjModel, n: int, delay: int) -> list[np.ndarray]:
    data = mujoco.MjData(model)
    per_env = (
        data.qpos,
        data.qvel,
        data.act,
        data.qacc_warmstart,
        np.float32(0.0),
        data.qfrc_applied,
        data.xfrc_applied,
        np.zeros(model.nu),
        np.full(model.nu, 0.5),
    )
    batch = [np.stack([np.asarray(x)] * n) for x in per_env]
    return [*batch, np.full(n, delay, np.int32)]


def test_cpu_physics_step_matches_mj_step(model: mujoco.MjModel) -> None:
    physics = s2s.CpuPhysics(model, threads=2)
    physics.set_arena(np.full(model.nhfielddata, 0.5, np.float32))
    out = physics.step(5, specs_for(model), *host_inputs(model, 3, delay=2))
    data = mujoco.MjData(model)
    model.hfield_data[:] = 0.5
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data, nstep=2)
    data.ctrl[:] = 0.5
    mujoco.mj_step(model, data, nstep=3)
    np.testing.assert_allclose(out["qpos"][2], data.qpos, atol=1e-6)
    np.testing.assert_allclose(out["geom_xpos"][1], data.geom_xpos, atol=1e-6)
    np.testing.assert_allclose(
        out["geom_xmat"][0], data.geom_xmat.reshape(-1, 3, 3), atol=1e-6
    )
    np.testing.assert_allclose(out["sensordata"][0], data.sensordata, atol=1e-5)
    assert physics.take_diverged() == 0


def test_cpu_physics_half_dt_runs_two_steps_per_substep(
    model: mujoco.MjModel,
) -> None:
    physics = s2s.CpuPhysics(model, threads=1)
    physics.use(s2s.half_dt(model, None))
    physics.set_arena(np.full(model.nhfielddata, 0.5, np.float32))
    out = physics.step(5, specs_for(model), *host_inputs(model, 1, delay=0))
    assert out["time"][0] == pytest.approx(5 * 0.004)


def test_cpu_physics_drops_diverged_env(model: mujoco.MjModel) -> None:
    physics = s2s.CpuPhysics(model, threads=1)
    physics.set_arena(np.full(model.nhfielddata, 0.5, np.float32))
    inputs = host_inputs(model, 1, delay=0)
    inputs[1][:] = 1e30
    out = physics.step(5, specs_for(model), *inputs)
    assert out["qpos"][0, 2] == s2s.DIVERGED_Z
    assert np.isfinite(out["xpos"]).all()
    assert physics.take_diverged() == 1
    assert physics.take_diverged() == 0


def test_cpu_physics_rejects_arena_of_another_model(model: mujoco.MjModel) -> None:
    physics = s2s.CpuPhysics(model, threads=1)
    with pytest.raises(ValueError, match="TileLayout.assemble"):
        physics.set_arena(np.full((2, model.nhfielddata), 0.5, np.float32))


def test_cpu_physics_loads_the_arena_into_every_thread(
    model: mujoco.MjModel,
) -> None:
    physics = s2s.CpuPhysics(model, threads=2)
    arena = np.linspace(0.0, 1.0, model.nhfielddata, dtype=np.float32)
    physics.set_arena(arena)
    physics.step(1, specs_for(model), *host_inputs(model, 2, delay=0))
    for m in physics.models:
        np.testing.assert_array_equal(m.hfield_data, arena)


def level_cfg(compose_cfg: Callable[..., Any], robot: str = "wojtek") -> Any:
    return compose_cfg(
        "sim2sim",
        "checkpoint=c",
        "name=n",
        "out_dir=o",
        "num_levels=8",
        "suite_instances=3",
        f"robot={robot}",
    )


def test_level_sets_names_and_sizes(compose_cfg: Callable[..., Any]) -> None:
    cfg = level_cfg(compose_cfg)
    config = hfield_config(cfg.robot.name, cfg.arena)
    sets = s2s.level_sets(cfg, config)
    cells = [f"suite/{t}_{k}" for t in SUITE_TERRAINS for k in (1, 2)]
    assert list(sets) == ["flat", "param/uniform", *cells]
    assert sets["param/uniform"].shape == (8, config.grid_size**2)
    assert all(sets[c].shape == (3, config.grid_size**2) for c in cells)
    np.testing.assert_allclose(sets["flat"], 0.5, atol=1e-6)


def test_uniform_levels_are_eval_brax_prefix(compose_cfg: Callable[..., Any]) -> None:
    cfg = level_cfg(compose_cfg)
    config = hfield_config(cfg.robot.name, cfg.arena)
    space = make_level_space("param", config, n_edits=1, edit_scale=0.0)
    eval_brax = space.uniform(np.random.default_rng(cfg.seed + 1), 2048)
    np.testing.assert_array_equal(
        s2s.level_sets(cfg, config)["param/uniform"], space.render(eval_brax[:8])
    )


@pytest.mark.parametrize("robot", ["wojtek", "go1"])
def test_suite_cells_match_eval_brax(
    compose_cfg: Callable[..., Any], robot: str
) -> None:
    cfg = level_cfg(compose_cfg, robot)
    config = hfield_config(cfg.robot.name, cfg.arena)
    rng = np.random.default_rng(cfg.seed)
    scale = ROBOT_SCALE[robot]
    eval_brax = {
        f"suite/{t}_{k}": np.stack(
            [suite_hfield(config, t, k, rng, scale) for _ in range(3)]
        )
        for t in SUITE_TERRAINS
        for k in SUITE_STRENGTHS
    }
    sets = s2s.level_sets(cfg, config)
    for name, hfields in sets.items():
        if name.startswith("suite/"):
            np.testing.assert_array_equal(hfields, eval_brax[name])


def test_summarize_reports_drop_against_mjx_match() -> None:
    fell = np.zeros((4, 2))
    travel = np.array([[2.0, 2.0], [2.0, 2.0], [0.5, 2.0], [0.5, 0.5]])
    worse = travel.copy()
    worse[0] = 0.0
    sets = {"flat": (fell, travel, 0), "suite/waves_1": (fell, travel, 0)}
    results = {
        "mjx_match": sets,
        "soft_contact": {"flat": (fell, worse, 2), "suite/waves_1": sets["flat"]},
    }
    metrics = s2s.summarize(results, threshold=1.0)
    assert metrics["mjx_match/flat/success_rate"] == pytest.approx(0.625)
    assert "mjx_match/flat/drop" not in metrics
    assert metrics["soft_contact/flat/drop"] == pytest.approx(0.25)
    assert metrics["soft_contact/suite/drop"] == 0.0
    assert metrics["soft_contact/flat/diverged_per_episode"] == pytest.approx(0.25)
    assert metrics["mjx_match/suite/waves_1/success_rate"] == pytest.approx(0.625)
