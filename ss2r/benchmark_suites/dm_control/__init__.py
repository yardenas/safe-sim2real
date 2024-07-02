import copy
from collections import OrderedDict

from gymnasium import Env
from dm_control.utils.rewards import tolerance
import numpy as np
from omegaconf import DictConfig
from ss2r.benchmark_suites.utils import get_domain_and_task

from ss2r.rl.types import EnvironmentFactory
from ss2r.rl.wrappers import ImageObservation


# From:
# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
def convert_dm_control_to_gym_space(dm_control_space):
    """Recursively convert dm_control_space into gym space.
    Note: Need to check the following cases of the input type, in the following
    order:
       (1) BoundedArraySpec
       (2) ArraySpec
       (3) OrderedDict.
    - Generally, dm_control observation_specs are OrderedDict with other spaces
      (e.g. ArraySpec) nested in it.
    - Generally, dm_control action_specs are of type `BoundedArraySpec`.
    To handle dm_control observation_specs as inputs, we check the following
    input types in order to enable recursive calling on each nested item.
    """
    from dm_env import specs
    from gymnasium import spaces

    if isinstance(dm_control_space, specs.BoundedArray):
        shape = dm_control_space.shape
        low = np.broadcast_to(dm_control_space.minimum, shape)
        high = np.broadcast_to(dm_control_space.maximum, shape)
        gym_box = spaces.Box(
            low=low, high=high, shape=None, dtype=dm_control_space.dtype
        )
        # Note: `gym.Box` doesn't allow both shape and min/max to be defined
        # at the same time. Thus we omit shape in the constructor and verify
        # that it's been implicitly set correctly.
        assert gym_box.shape == dm_control_space.shape, (
            gym_box.shape,
            dm_control_space.shape,
        )
        return gym_box
    elif isinstance(dm_control_space, specs.Array):
        if isinstance(dm_control_space, specs.BoundedArray):
            raise ValueError("The order of the if-statements matters.")
        bound = lambda dtype, val, alt: val if np.issubdtype(dtype, np.bool_) else alt
        return spaces.Box(
            low=bound(dm_control_space.dtype, 0.0, -float("inf")),
            high=bound(dm_control_space.dtype, 1.0, float("inf")),
            shape=(
                dm_control_space.shape
                if (
                    len(dm_control_space.shape) == 1
                    or (
                        len(dm_control_space.shape) == 3
                        and np.issubdtype(dm_control_space.dtype, np.integer)
                    )
                )
                else (int(np.prod(dm_control_space.shape)),)
            ),
            dtype=dm_control_space.dtype,
        )
    elif isinstance(dm_control_space, OrderedDict):
        return spaces.Dict(
            OrderedDict(
                [
                    (key, convert_dm_control_to_gym_space(value))
                    for key, value in dm_control_space.items()
                ]
            )
        )
    else:
        raise ValueError(dm_control_space)


# Modified from:
# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/environments/adapters/dm_control_adapter.py
class DMCWrapper:
    """Wrapper to convert DeepMind Control tasks to OpenAI Gym format."""

    spec: None = None

    def __init__(self, domain_name, task_name):
        """Initializes DeepMind Control tasks.

        Args:
            domain_name (str): name of DeepMind Control domain
            task_name (str): name of DeepMind Control task
        """

        # Supports tasks in DeepMind Control Suite
        from dm_control import suite

        if domain_name.startswith("dm_"):
            domain_name = domain_name.replace("dm_", "")
        env = suite.load(domain_name=domain_name, task_name=task_name)
        self._setup(env)

    def _setup(self, env, exclude_keys=[]):
        """Sets up environment and corresponding spaces.

        Args:
            env (object): DeepMind Control Suite environment
            exclude_keys (list): list of keys to exclude from observation
        """
        from dm_control.suite.wrappers import action_scale
        from dm_env import specs

        assert isinstance(env.observation_spec(), OrderedDict)
        assert isinstance(env.action_spec(), specs.BoundedArray)
        env = action_scale.Wrapper(
            env,
            minimum=np.ones_like(env.action_spec().minimum) * -1,
            maximum=np.ones_like(env.action_spec().maximum),
        )
        np.testing.assert_equal(env.action_spec().minimum, -1)
        np.testing.assert_equal(env.action_spec().maximum, 1)
        self.env = env
        # Can remove parts of observation by excluding keys here
        observation_keys = tuple(env.observation_spec().keys())
        self.observation_keys = tuple(
            key for key in observation_keys if key not in exclude_keys
        )
        observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.observation_space = type(observation_space)(
            [
                (name, copy.deepcopy(space))
                for name, space in observation_space.spaces.items()
                if name in self.observation_keys
            ]
        )
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())
        if len(self.action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(self.action_space)
            )

    def _filter_observation(self, observation):
        """Filters excluded keys from observation."""
        observation = type(observation)(
            (
                (name, np.reshape(value, self.observation_space.spaces[name].shape))
                for name, value in observation.items()
                if name in self.observation_keys
            )
        )
        return observation

    def step(self, a):
        """Takes step in environment.

        Args:
            a (np.ndarray): action

        Returns:
            s (np.ndarray): flattened next state
            r (float): reward
            t (bool): terminal flag
            truncated (bool): truncated flag
            info (dict): dictionary with additional environment info
        """
        time_step = self.env.step(a)
        r = time_step.reward or 0.0
        d = time_step.last()
        info = {
            key: value
            for key, value in time_step.observation.items()
            if key not in self.observation_keys
        }
        observation = self._filter_observation(time_step.observation)
        return observation, r, False, d, info

    def reset(self, *, seed=None, options=None):
        """Resets environment and returns flattened initial state."""
        time_step = self.env.reset()
        if seed is not None:
            self.seed(seed)
        observation = self._filter_observation(time_step.observation)
        return observation, {}

    def seed(self, seed):
        self.env.task._random = np.random.RandomState(seed)

    def render(self, camera_id=None, **kwargs):
        self.env.task.visualize_reward = kwargs.get("visualize_reward", False)
        if camera_id is None:
            camera_id = -1
        return self.env.physics.render(camera_id=camera_id)

    @property
    def np_random(self):
        return self.env.task._random


class ActionCostWrapper:
    def __init__(self, env: Env, cost_multiplier: float = 0):
        self.env = env
        self.cost_multiplier = cost_multiplier

    def step(self, action):
        action_cost = (
            self.cost_multiplier * (1 - tolerance(action, (-0.1, 0.1), 0.1))[0]
        )
        observation, reward, terminal, truncated, info = self.env.step(action)
        return observation, reward - action_cost, terminal, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class ConstraintWrapper:
    def __init__(self, env: Env, slider_position_bound: float):
        self.env = env
        self.physics = env.env.physics
        self.slider_position_bound = slider_position_bound

    def step(self, action):
        observation, reward, terminal, truncated, info = self.env.step(action)
        slider_pos = self.physics.cart_position().copy()
        cost = float(np.abs(slider_pos) >= self.slider_position_bound)
        info["cost"] = cost
        return observation, reward, terminal, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        domain_name, task_cfg = get_domain_and_task(cfg)
        if task_cfg.task in ["swingup_sparse_hard", "safe_swingup_sparse_hard"]:
            task = "swingup_sparse"
        else:
            task = task_cfg.task
        env = DMCWrapper(domain_name, task)
        if "safe" in task_cfg.task:
            env = ConstraintWrapper(env, task_cfg.slider_position_bound)
        if task_cfg.task in ["swingup_sparse_hard", "safe_swingup_sparse_hard"]:
            env = ActionCostWrapper(env, cost_multiplier=task_cfg.cost_multiplier)
        if task_cfg.image_observation.enabled:
            env = ImageObservation(
                env,
                task_cfg.image_observation.image_size,
                task_cfg.image_observation.image_format,
                render_kwargs={
                    "visualize_reward": task_cfg.image_observation.visualize_reward,
                    "camera_id": 0,
                },
            )
        else:
            from gymnasium.wrappers.flatten_observation import FlattenObservation

            env = FlattenObservation(env)
        return env

    return make_env


ENVIRONMENTS = {
    ("dm_cartpole", "balance"),
    ("dm_cartpole", "swingup"),
    ("dm_cartpole", "swingup_sparse"),
    ("dm_cartpole", "swingup_sparse_hard"),
    ("dm_cartpole", "safe_swingup_sparse_hard"),
    ("dm_humanoid", "stand"),
    ("dm_humanoid", "walk"),
    ("dm_manipulator", "bring_ball"),
    ("dm_manipulator", "bring_peg"),
    ("dm_manipulator", "insert_ball"),
    ("dm_manipulator", "insert_peg"),
    ("dm_quadruped", "walk"),
    ("dm_quadruped", "run"),
    ("dm_walker", "stand"),
    ("dm_walker", "walk"),
}
