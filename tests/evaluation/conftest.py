import mujoco
import pytest

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


@pytest.fixture
def robot_xml() -> str:
    return ROBOT_XML
