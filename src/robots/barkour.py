"""Google Barkour vB: Playground's Barkour joystick env on the tiled arena."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.barkour import joystick

from config.hfield_config import HFieldConfig
from robots.terrain import TerrainRobot, arena_scene, compile_scene

HOME_POSE = "0 0.5 1.0 0 0.5 1.0 0 0.5 1.0 0 0.5 1.0"
FEET = (
    "foot_front_left",
    "foot_hind_left",
    "foot_front_right",
    "foot_hind_right",
)
FEET_SENSORS = "\n".join(
    f'    <contact name="{foot}_floor_found" geom1="{foot}" geom2="floor"'
    ' reduce="mindist" num="1" data="found"/>'
    for foot in FEET
)


class BarkourTerrain(TerrainRobot, joystick.Joystick):
    # standing trunk height 0.253 m (measured) over Wojtek's design 0.10 m
    SCALE = 2.5
    STANDING_HEIGHT = 0.253

    @classmethod
    def scene_xml(cls, config: HFieldConfig) -> str:
        return arena_scene(
            "barkour",
            "barkour_vb_mjx.xml",
            config,
            sensors=f"<sensor>\n{FEET_SENSORS}\n  </sensor>",
            spawn_z=0.33,
            home_pose=HOME_POSE,
        )

    @classmethod
    def assets(cls) -> dict[str, bytes]:
        return joystick.get_assets()

    @staticmethod
    def default_config() -> Any:
        return joystick.default_config()

    def native_command_range(self) -> tuple[list[float], list[float]]:
        # barkour/joystick.py: lin_vel_x, lin_vel_y, ang_vel_yaw
        c = self._config
        ranges = (c.lin_vel_x, c.lin_vel_y, c.ang_vel_yaw)
        return [float(r[0]) for r in ranges], [float(r[1]) for r in ranges]

    def _build_env(self, config: Any, hfield: HFieldConfig) -> None:
        mjx_env.MjxEnv.__init__(self, config, None)
        self._xml_path = self._write_scene(hfield)
        self._model_assets = self.assets()
        model = compile_scene(self.scene_xml(hfield), self._model_assets)
        model.vis.global_.offwidth = 3840
        model.vis.global_.offheight = 2160
        model.opt.timestep = config.sim_dt
        self._mj_model = model
        self._mjx_model = mjx.put_model(model, impl=config.impl)
        self._feet_floor_found_sensor = [f"{foot}_floor_found" for foot in FEET]
        self._post_init()

    def actor_obs(self, state: mjx_env.State) -> jax.Array:
        return self._get_obs(
            state.data, dict(state.info), state.obs["state"], state.info["rng"]
        )

    def _gravity_z(self, data: mjx.Data) -> jax.Array:
        # Playground's Barkour "gravity" is the trunk's up vector: +1 upright
        return -self._get_gravity(data)[-1]

    def _obs_dict(self, obs: Any) -> dict[str, jax.Array]:
        return {"state": obs, "privileged_state": obs}

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Playground's `step` with its termination swapped for
        `_get_termination` (its `torso_z < 0.18` reads the world frame, which
        a pit tile or a raised level breaks) plus its joint limits."""
        rng, cmd_rng, noise_rng, pert_rng = jax.random.split(
            state.info["rng"], 4
        )
        state = self._maybe_apply_perturbation(state, pert_rng)

        motor_targets = self._default_pose + action * self._config.action_scale
        motor_targets = jnp.clip(motor_targets, self._lowers, self._uppers)
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        obs = self._get_obs(data, state.info, state.obs["state"], noise_rng)
        joint_angles = data.qpos[7:]
        contact = jnp.array(
            [
                data.sensordata[
                    self._mj_model.sensor_adr[self._mj_model.sensor(s).id]
                ]
                > 0
                for s in self._feet_floor_found_sensor
            ]
        )
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt

        done = self._get_termination(data)
        done |= jnp.any(joint_angles < self._lowers)
        done |= jnp.any(joint_angles > self._uppers)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact
        )
        scales = self._config.reward_config.scales
        rewards = {k: v * scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["last_act"] = action
        state.info["last_vel"] = data.qvel[6:]
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["command"] = jnp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jnp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state = dataclasses.replace(
            state, data=data, obs=obs, reward=reward, done=jnp.float32(done)
        )
        return self._with_scan(state)
