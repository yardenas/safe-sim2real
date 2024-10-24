import functools
from typing import Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util as jtu
from brax.envs.base import Env, State
from omegaconf import OmegaConf

from ss2r.benchmark_suites import rewards
from ss2r.benchmark_suites.rccar.model import CarParams, RaceCarDynamics

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(
    jnp.array([-4.5, -4.5, -4.0, -2.5, -2.5, -1.0])
)


def domain_randomization(sys, rng, cfg):
    def sample_from_bounds(value, key):
        """
        Sample from a JAX uniform distribution if the value is a list of two elements.
        """
        if isinstance(value, list) and len(value) == 2:
            lower, upper = value
            # Sample from jax.random.uniform with the given key
            return jax.random.uniform(key, shape=(), minval=lower, maxval=upper)
        return value

    @jax.vmap
    def randomize(rng):
        bounds = CarParams(**cfg)
        # Define a custom tree structure that treats lists as leaves
        treedef = jtu.tree_structure(bounds, is_leaf=lambda x: isinstance(x, list))
        # Generate random keys only for the relevant leaves (i.e., lists with 2 elements)
        keys = jax.random.split(rng, num=treedef.num_leaves)
        # Rebuild the tree with the keys, only where there are valid leaves
        keys = jtu.tree_unflatten(treedef, keys)
        # Map over the tree, generating random values where needed
        sys = jtu.tree_map(
            sample_from_bounds, bounds, keys, is_leaf=lambda x: isinstance(x, list)
        )
        return sys, jax.flatten_util.ravel_pytree(sys)[0]

    cfg = OmegaConf.to_container(cfg)
    in_axes = jax.tree_map(lambda _: 0, sys)
    sys, params = randomize(rng)
    return sys, in_axes, params[:, None]


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = (
        state[..., 0:1],
        state[..., 3 + int(encode_angle) : 4 + int(encode_angle)],
    )
    y_pos, y_vel = (
        state[..., 1:2],
        state[:, 4 + int(encode_angle) : 5 + int(encode_angle)],
    )
    theta = state[..., 2 : 3 + int(encode_angle)]
    new_state = jnp.concatenate(
        [y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle) :]],
        axis=-1,
    )
    assert state.shape == new_state.shape
    return new_state


def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Encodes the angle (theta) as sin(theta) ant cos(theta)"""
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx : angle_idx + 1]
    state_encoded = jnp.concatenate(
        [
            state[..., :angle_idx],
            jnp.sin(theta),
            jnp.cos(theta),
            state[..., angle_idx + 1 :],
        ],
        axis=-1,
    )
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded


def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """Decodes the angle (theta) from sin(theta) ant cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(
        state[..., angle_idx : angle_idx + 1], state[..., angle_idx + 1 : angle_idx + 2]
    )
    state_decoded = jnp.concatenate(
        [state[..., :angle_idx], theta, state[..., angle_idx + 2 :]], axis=-1
    )
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


class RCCarEnvReward:
    def __init__(
        self,
        goal: jnp.array,
        encode_angle: bool = False,
        ctrl_cost_weight: float = 0.005,
        bound: float = 0.1,
        margin_factor: float = 10.0,
    ):
        self._angle_idx = 2
        self.dim_action = (2,)
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        self.tolerance_reward = functools.partial(
            rewards.tolerance,
            bounds=(0.0, bound),
            margin=margin_factor * bound,
            value_at_margin=0.1,
            sigmoid="long_tail",
        )

    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """Computes the reward for the given transition"""
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> jnp.array:
        """Computes the reward/penalty for the given action"""
        return -(action**2).sum(-1)

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """Computes the reward for the given observations"""
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist**2 + theta_dist**2)
        reward = self.tolerance_reward(total_dist)
        return reward


class RCCar(Env):
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])
    _init_pose: jnp.array = jnp.array([1.42, -1.04, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = OBS_NOISE_STD_SIM_CAR

    def __init__(
        self,
        car_model_params: dict,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = True,
        use_obs_noise: bool = False,
        margin_factor: float = 10.0,
        max_throttle: float = 1.0,
        dt: float = 1 / 30.0,
    ):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        self._dt = dt
        self.dim_state = (7,) if encode_angle else (6,)
        self.encode_angle = encode_angle
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.dynamics_model = RaceCarDynamics(dt=self._dt)
        self.sys = CarParams(**car_model_params)
        self.use_obs_noise = use_obs_noise
        self.reward_model = RCCarEnvReward(
            goal=self._goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            margin_factor=margin_factor,
        )

    def _obs(self, state: jnp.array, rng: jax.random.PRNGKey) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(
                rng, shape=self.dim_state
            )
        else:
            obs = state
        # encode angle to sin(theta) ant cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to a random initial state close to the initial pose"""

        # sample random initial state
        key_pos, key_vel, key_obs = jax.random.split(rng, 3)
        init_pos = self._init_pose[:2] + jax.random.uniform(
            key_pos, shape=(2,), minval=-0.10, maxval=0.10
        )
        init_theta = self._init_pose[2:] + jax.random.uniform(
            key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
        )
        init_vel = jnp.zeros((3,)) + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])
        init_state = self._obs(init_state, rng=key_obs)
        return State(
            pipeline_state=None,
            obs=init_state,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
        )

    def step(self, state: State, action: jax.Array) -> State:
        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])
        obs = state.obs
        if self.encode_angle:
            dynamics_state = decode_angles(obs, self._angle_idx)
        next_dynamics_state = self.dynamics_model.step(dynamics_state, action, self.sys)
        # FIXME (yarden): hard-coded key is bad here.
        next_obs = self._obs(next_dynamics_state, rng=jax.random.PRNGKey(0))
        reward = self.reward_model.forward(obs=None, action=action, next_obs=next_obs)
        done = jnp.asarray(0.0)
        next_state = State(
            pipeline_state=state.pipeline_state,
            obs=next_obs,
            reward=reward,
            done=done,
            metrics=state.metrics,
            info=state.info,
        )
        return next_state

    @property
    def dt(self):
        return self._dt

    @property
    def observation_size(self) -> int:
        if self.encode_angle:
            return 7
        else:
            return 6

    @property
    def action_size(self) -> int:
        # [steering, throttle]
        return 2

    def backend(self) -> str:
        return "positional"
