import jax
import mujoco
import numpy as np
import pytest

from evaluation import cpu_physics as cp
from evaluation.physics_variants import half_dt, mjx_match


def test_sensor_index_follows_names(model: mujoco.MjModel, robot_xml: str) -> None:
    spec = mujoco.MjSpec.from_string(robot_xml)
    spec.add_sensor(
        name="extra",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="foot",
    )
    variant = spec.compile()
    np.testing.assert_array_equal(
        cp.sensor_index(model, variant), np.arange(model.nsensordata)
    )
    reordered = mujoco.MjSpec.from_string(robot_xml)
    reordered.delete(reordered.sensors[0])
    reordered.add_sensor(
        name="foot_pos",
        type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname="foot",
    )
    np.testing.assert_array_equal(
        cp.sensor_index(model, reordered.compile()), [3, 4, 5, 0, 1, 2]
    )


def test_sensor_index_rejects_missing_sensor(
    model: mujoco.MjModel, robot_xml: str
) -> None:
    spec = mujoco.MjSpec.from_string(robot_xml)
    spec.delete(spec.sensors[1])
    with pytest.raises(ValueError, match="gyro"):
        cp.sensor_index(model, spec.compile())


def test_cpu_physics_rejects_incompatible_models(
    model: mujoco.MjModel, robot_xml: str
) -> None:
    physics = cp.CpuPhysics(model, threads=1)
    other = mujoco.MjModel.from_xml_string(robot_xml.replace("<freejoint/>", ""))
    with pytest.raises(ValueError, match="nq"):
        physics.use(other)
    odd = mjx_match(model, None)
    odd.opt.timestep = 0.003
    with pytest.raises(ValueError, match="divide"):
        physics.use(odd)


def specs_for(model: mujoco.MjModel) -> dict[str, jax.ShapeDtypeStruct]:
    data = mujoco.MjData(model)
    shapes = {f: np.shape(getattr(data, f)) for f in cp.OUT_FIELDS}
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
    physics = cp.CpuPhysics(model, threads=2)
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
    physics = cp.CpuPhysics(model, threads=1)
    physics.use(half_dt(model, None))
    physics.set_arena(np.full(model.nhfielddata, 0.5, np.float32))
    out = physics.step(5, specs_for(model), *host_inputs(model, 1, delay=0))
    assert out["time"][0] == pytest.approx(5 * 0.004)


def test_cpu_physics_drops_diverged_env(model: mujoco.MjModel) -> None:
    physics = cp.CpuPhysics(model, threads=1)
    physics.set_arena(np.full(model.nhfielddata, 0.5, np.float32))
    inputs = host_inputs(model, 1, delay=0)
    inputs[1][:] = 1e30
    out = physics.step(5, specs_for(model), *inputs)
    assert out["qpos"][0, 2] == cp.DIVERGED_Z
    assert np.isfinite(out["xpos"]).all()
    assert physics.take_diverged() == 1
    assert physics.take_diverged() == 0


def test_cpu_physics_rejects_arena_of_another_model(model: mujoco.MjModel) -> None:
    physics = cp.CpuPhysics(model, threads=1)
    with pytest.raises(ValueError, match="TileLayout.assemble"):
        physics.set_arena(np.full((2, model.nhfielddata), 0.5, np.float32))


def test_cpu_physics_loads_the_arena_into_every_thread(
    model: mujoco.MjModel,
) -> None:
    physics = cp.CpuPhysics(model, threads=2)
    arena = np.linspace(0.0, 1.0, model.nhfielddata, dtype=np.float32)
    physics.set_arena(arena)
    physics.step(1, specs_for(model), *host_inputs(model, 2, delay=0))
    for m in physics.models:
        np.testing.assert_array_equal(m.hfield_data, arena)
