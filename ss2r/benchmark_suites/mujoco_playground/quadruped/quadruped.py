"""Quadruped environment."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "quadruped.xml"

# Heights and speeds for rewards
_STAND_HEIGHT = 0.4
WALK_SPEED = 1.0
RUN_SPEED = 3.0


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
        move_speed: float,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError("Vision not implemented for Quadruped.")

        self._move_speed = move_speed
        self._stand_or_move_reward = (
            self._move_reward if move_speed > 0 else self._stand_reward
        )

        self._xml_path = _XML_PATH.as_posix()
        self._mj_model = mujoco.MjModel.from_xml_string(
            _XML_PATH.read_text(), common.get_assets()
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._post_init()

    def _post_init(self) -> None:
        self._torso_body_id = self.mj_model.body("torso").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        data = mjx_env.init(self.mjx_model)
        metrics = {"reward/standing": jp.zeros(()), "reward/move": jp.zeros(())}
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
        return jp.concatenate(
            [
                self._joint_angles(data),
                self._torso_height(data).reshape(1),
                self._center_of_mass_velocity(data),
                data.qvel,
            ]
        )

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
    ) -> jax.Array:
        del info
        standing = reward.tolerance(
            self._torso_height(data),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        metrics["reward/standing"] = standing
        move_reward = self._stand_or_move_reward(data)
        metrics["reward/move"] = move_reward
        return standing * move_reward

    def _stand_reward(self, data: mjx.Data) -> jax.Array:
        return reward.tolerance(self._center_of_mass_velocity(data), margin=1).mean()

    def _move_reward(self, data: mjx.Data) -> jax.Array:
        speed = jp.linalg.norm(self._center_of_mass_velocity(data)[:2])
        return reward.tolerance(
            speed, bounds=(self._move_speed, float("inf")), margin=self._move_speed
        )

    def _joint_angles(self, data: mjx.Data) -> jax.Array:
        return data.qpos[7:]

    def _torso_height(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self._torso_body_id, -1]

    def _center_of_mass_velocity(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, "torso_subtreelinvel")

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
