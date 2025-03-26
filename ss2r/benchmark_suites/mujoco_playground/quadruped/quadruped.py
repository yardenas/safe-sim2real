"""Quadruped environment."""

from functools import partial
from itertools import product
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from brax.envs import Wrapper
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env, reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = epath.Path(__file__).parent / "quadruped.xml"

WALK_SPEED = 0.5
RUN_SPEED = 5.0

_TORSO_ID = 1
_TOE_IDS = [7, 11, 15, 19]
_JOINTS_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        rng, rng_ = jax.random.split(rng)
        friction = jax.random.uniform(
            rng_, minval=cfg.friction[0], maxval=cfg.friction[1]
        )
        friction_sample = sys.geom_friction.copy()
        friction_sample = friction_sample.at[0, 0].add(friction)
        # Toes have a default friction coefficient of 1.5
        friction_sample = friction_sample.at[jp.asarray(_TOE_IDS), 0].add(friction)
        rng, rng_ = jax.random.split(rng)
        torso_density_sample = jax.random.uniform(
            rng_, minval=cfg.torso[0], maxval=cfg.torso[1]
        )
        # Default density of 1000.
        scale = (torso_density_sample + 1000) / 1000.0
        mass = sys.body_mass.at[_TORSO_ID].multiply(scale)
        inertia = sys.body_inertia.at[_TORSO_ID].multiply(scale**3)
        rng, rng_ = jax.random.split(rng)
        damping_sample = jax.random.uniform(
            rng_, minval=cfg.damping[0], maxval=cfg.friction[1]
        )
        damping = sys.dof_damping.copy()
        damping = damping.at[jp.asarray(_JOINTS_IDS)].add(damping_sample)
        return (
            friction_sample,
            mass,
            inertia,
            damping,
            jp.stack(
                [friction, torso_density_sample, damping],
                axis=-1,
            ),
        )

    friction_sample, mass, inertia, damping, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {"geom_friction": 0, "body_inertia": 0, "body_mass": 0, "dof_damping": 0}
    )
    sys = sys.tree_replace(
        {
            "geom_friction": friction_sample,
            "body_inertia": inertia,
            "body_mass": mass,
            "dof_damping": damping,
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


def _find_non_contacting_height(mjx_model, data, orientation, x_pos=0.0, y_pos=0.0):
    def body_fn(state):
        z_pos, num_contacts, num_attempts, _ = state
        qpos = data.qpos.at[:3].set(jp.array([x_pos, y_pos, z_pos]))
        qpos = qpos.at[3:7].set(jp.array(orientation))
        ndata = data.replace(qpos=qpos)
        ndata = mjx.forward(mjx_model, ndata)
        num_contacts = ndata.ncon
        z_pos += 0.01
        num_attempts += 1
        return (z_pos, num_contacts, num_attempts, ndata)

    initial_state = (0.0, 1, 0, data)  # (z_pos, num_contacts, num_attempts)
    *_, num_attemps, ndata = jax.lax.while_loop(
        lambda state: jp.greater(state[1], 0) & jp.less_equal(state[2], 10000),
        body_fn,
        initial_state,
    )
    ndata = jax.tree_map(
        lambda x, y: jp.where(jp.less(num_attemps, 10000), x, y), ndata, data
    )
    return ndata


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
        lower, upper = (
            self._mj_model.actuator_ctrlrange[:, 0],
            self._mj_model.actuator_ctrlrange[:, 1],
        )
        action = (action + 1.0) / 2.0 * (upper - lower) + lower
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


class ConstraintWrapper(Wrapper):
    def __init__(self, env: MjxEnv, angle_tolerance: float):
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jp.pi / 180.0
        joint_names = [
            "yaw_front_left",
            "pitch_front_left",
            "knee_front_left",
            "ankle_front_left",
            "toe_front_left",
            "yaw_front_right",
            "pitch_front_right",
            "knee_front_right",
            "ankle_front_right",
            "toe_front_right",
            "yaw_back_right",
            "pitch_back_right",
            "knee_back_right",
            "ankle_back_right",
            "toe_back_right",
            "yaw_back_left",
            "pitch_back_left",
            "knee_back_left",
            "ankle_back_left",
            "toe_back_left",
        ]
        joint_ids = jp.asarray(
            [
                mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)
                for name in joint_names
            ]
        )
        self.joint_ranges = [env.mj_model.jnt_range[id_] for id_ in joint_ids]
        self.qpos_ids = jp.asarray([env.mj_model.jnt_qposadr[id_] for id_ in joint_ids])

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        cost = jp.zeros_like(nstate.reward)
        for qpos_id, joint_range in zip(self.qpos_ids, self.joint_ranges):
            angle = nstate.data.qpos[qpos_id]
            normalized_angle = normalize_angle(angle)
            lower_limit = normalize_angle(joint_range[0] - self.angle_tolerance)
            upper_limit = normalize_angle(joint_range[1] + self.angle_tolerance)
            is_out_of_range_case1 = (normalized_angle < lower_limit) & (
                normalized_angle >= upper_limit
            )
            is_out_of_range_case2 = (normalized_angle < lower_limit) | (
                normalized_angle >= upper_limit
            )
            out_of_range = jp.where(
                upper_limit < lower_limit, is_out_of_range_case1, is_out_of_range_case2
            )
            cost += out_of_range
        nstate.info["cost"] = (cost > 0).astype(jp.float32)
        return nstate


def normalize_angle(angle, lower_bound=-jp.pi, upper_bound=jp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


dm_control_suite.register_environment(
    "QuadrupedWalk", partial(Quadruped, desired_speed=WALK_SPEED), default_config
)
dm_control_suite.register_environment(
    "QuadrupedRun", partial(Quadruped, desired_speed=RUN_SPEED), default_config
)

for run in [True, False]:

    def make(run, **kwargs):
        run_str = "Run" if run else "Walk"
        limit = kwargs["config"]["angle_tolerance"]
        env = dm_control_suite.load(f"Quadruped{run_str}", **kwargs)
        env = ConstraintWrapper(env, limit)
        return env

    run_str = "Run" if run else "Walk"
    name_str = f"SafeQuadruped{run_str}"
    dm_control_suite.register_environment(
        name_str, partial(make, run=run), dm_control_suite.walker.default_config
    )
