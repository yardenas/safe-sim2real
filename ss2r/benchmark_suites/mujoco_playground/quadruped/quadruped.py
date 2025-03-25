"""Quadruped environment."""

from functools import partial
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "quadruped.xml"

WALK_SPEED = 0.5
RUN_SPEED = 5.0


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )


class Quadruped(mjx_env.MjxEnv):
    """Quadruped environment."""

    def __init__(
        self,
        desired_speed: float,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError("Vision not implemented for Quadruped.")

        self._desired_speed = desired_speed
        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), common.get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()
        self._hinge_ids = jp.nonzero(
            self._mj_model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
        )
        self._imu_sensor_ids = jp.where(
            jp.in1d(
                self._mj_model.sensor_type,
                (
                    mujoco.mjtSensor.mjSENS_GYRO.value,
                    mujoco.mjtSensor.mjSENS_ACCELEROMETER.value,
                ),
            )
        )
        self._force_torque_ids = jp.where(
            jp.in1d(
                self._mj_model.sensor_type,
                (
                    mujoco.mjtSensor.mjSENS_FORCE.value,
                    mujoco.mjtSensor.mjSENS_TORQUE.value,
                ),
            )
        )

    def _post_init(self) -> None:
        self._torso_body_id = self.mj_model.body("torso").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        metrics = {"reward/upright": jp.zeros(()), "reward/move": jp.zeros(())}
        info = {"rng": rng}
        reward, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        reward = self._get_reward(data, action, state.info, state.metrics)
        obs = self._get_obs(data, state.info)
        done = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        del info
        ego = self._egocentric_state(data)
        torso_vel = self.torso_velocity(data)
        upright = self.torso_upright(data)
        imu = self.imu(data)
        force_torque = self.force_torque(data)
        return jp.hstack((ego, torso_vel, upright, imu, force_torque))

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del info
        move_reward = reward.tolerance(
            self.torso_velocity(data)[0],
            bounds=(self._desired_speed, float("inf")),
            sigmoid="linear",
            margin=self._desired_speed,
            value_at_margin=0.5,
        )
        upright_reward = self._upright_reward(data)
        metrics["reward/move"] = move_reward
        metrics["reward/upright"] = upright_reward
        return move_reward * upright_reward

    def _upright_reward(self, data: mjx.Data, deviation_angle: float = 0) -> jax.Array:
        deviation = jp.cos(jp.deg2rad(deviation_angle))
        upright = self.torso_upright(data)
        return reward.tolerance(
            upright,
            bounds=(deviation, float("inf")),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )

    def _egocentric_state(self, data: mjx.Data) -> jax.Array:
        return jp.hstack(
            (data.qpos[self._hinge_ids], data.qvel[self._hinge_ids], data.act)
        )

    def torso_upright(self, data: mjx.Data) -> jax.Array:
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        return jp.asarray(data.xmat["torso", "zz"])

    def torso_velocity(self, data: mjx.Data) -> jax.Array:
        return data.sensordata["velocimeter"]

    def imu(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._imu_sensor_ids]

    def force_torque(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._force_torque_ids]

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


dm_control_suite.register_environment(
    "QuadrupedWalk", partial(Quadruped, desired_speed=WALK_SPEED), default_config
)
dm_control_suite.register_environment(
    "QuadrupedRun", partial(Quadruped, RUN_SPEED), default_config
)
