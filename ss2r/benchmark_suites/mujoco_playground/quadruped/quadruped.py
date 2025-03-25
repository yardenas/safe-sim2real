"""Quadruped environment."""

from functools import partial
from itertools import product
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import dm_control_suite
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = epath.Path(__file__).parent / "quadruped.xml"

WALK_SPEED = 0.5
RUN_SPEED = 5.0


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        rng, rng_ = jax.random.split(rng)
        friction = jax.random.uniform(
            rng_, minval=cfg.friction[0], maxval=cfg.friction[1]
        )
        friction_sample = sys.geom_friction.copy()
        friction_sample = friction_sample.at[0, 0].add(friction)
        friction_sample = jp.clip(friction_sample, a_min=0.0, a_max=1.0)
        rng = jax.random.split(rng, 8)
        # Ensure symmetry
        names_ids = {
            k: mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, k)
            for k in [
                "right_hip_x",
                "left_hip_x",
                "right_hip_y",
                "left_hip_y",
                "right_hip_z",
                "left_hip_z",
                "left_knee",
                "right_knee",
            ]
        }
        gain_sample = sys.actuator.gain.copy()
        gain_hip_x = jax.random.uniform(
            rng[0], minval=cfg.gain_hip.x[0], maxval=cfg.gain_hip.x[1]
        )
        gain_hip_y = jax.random.uniform(
            rng[1], minval=cfg.gain_hip.y[0], maxval=cfg.gain_hip.y[1]
        )
        gain_hip_z = jax.random.uniform(
            rng[2], minval=cfg.gain_hip.z[0], maxval=cfg.gain_hip.z[1]
        )
        gain_knee = jax.random.uniform(
            rng[3], minval=cfg.gain_knee[0], maxval=cfg.gain_knee[1]
        )
        gear_sample = sys.actuator.gear.copy()
        gear_hip_x = jax.random.uniform(
            rng[4], minval=cfg.gear_hip.x[0], maxval=cfg.gear_hip.x[1]
        )
        gear_hip_y = jax.random.uniform(
            rng[5], minval=cfg.gear_hip.y[0], maxval=cfg.gear_hip.y[1]
        )
        gear_hip_z = jax.random.uniform(
            rng[6], minval=cfg.gear_hip.z[0], maxval=cfg.gear_hip.z[1]
        )
        gear_knee = jax.random.uniform(
            rng[7], minval=cfg.gear_knee[0], maxval=cfg.gear_knee[1]
        )
        name_values = {
            "right_hip_x": (gain_hip_x, gear_hip_x),
            "left_hip_x": (gain_hip_x, gear_hip_x),
            "right_hip_y": (gain_hip_y, gear_hip_y),
            "left_hip_y": (gain_hip_y, gear_hip_y),
            "right_hip_z": (gain_hip_z, gear_hip_z),
            "left_hip_z": (gain_hip_z, gear_hip_z),
            "left_knee": (gain_knee, gear_knee),
            "right_knee": (gain_knee, gear_knee),
        }
        for name, (gain, gear) in name_values.items():
            actuator_id = names_ids[name]
            gear_sample = gear_sample.at[actuator_id].add(gear)
            gain_sample = gain_sample.at[actuator_id].add(gain)
        return (
            friction_sample,
            gear_sample,
            gain_sample,
            jp.stack(
                [
                    friction,
                    gain_hip_x,
                    gain_hip_y,
                    gain_hip_z,
                    gain_knee,
                    gear_hip_x,
                    gear_hip_y,
                    gear_hip_z,
                    gear_knee,
                ],
                axis=-1,
            ),
        )

    friction_sample, gear_sample, gain_sample, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator.gear": 0,
            "actuator.gain": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_friction": friction_sample,
            "actuator.gear": gear_sample,
            "actuator.gain": gain_sample,
        }
    )
    return sys, in_axes, samples


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

    def _post_init(self):
        # TODO (yarden): not 100% that this is correct.
        self._force_torque_names = [
            f"{f}_toe_{pos}_{side}"
            for (f, pos, side) in product(
                ("force", "torque"), ("front", "back"), ("left", "right")
            )
        ]
        self._torso_id = self._mj_model.body("torso").id

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
        del info, action
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

    def _upright_reward(self, data: mjx.Data) -> jax.Array:
        upright = self.torso_upright(data)
        return reward.tolerance(
            upright,
            bounds=(1, float("inf")),
            sigmoid="linear",
            margin=2,
            value_at_margin=0,
        )

    def _egocentric_state(self, data: mjx.Data) -> jax.Array:
        return jp.hstack((data.qpos[7:], data.qvel[7:], data.act))

    def torso_upright(self, data: mjx.Data) -> jax.Array:
        return data.xmat[self._torso_id, 2, 2]

    def torso_velocity(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "velocimeter")

    def imu(self, data: mjx.Data) -> jax.Array:
        gyro = mjx_env.get_sensor_data(self.mj_model, data, "imu_gyro")
        accelerometer = mjx_env.get_sensor_data(self.mj_model, data, "imu_accel")
        return jp.hstack((gyro, accelerometer))

    def force_torque(self, data: mjx.Data) -> jax.Array:
        return jp.hstack(
            [
                mjx_env.get_sensor_data(self.mj_model, data, name)
                for name in self._force_torque_names
            ]
        )

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
