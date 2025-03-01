from typing import Callable, Optional, Tuple

import jax
import numpy as np
from brax.envs.wrappers import training as brax_training
from mujoco import mjx
from mujoco_playground import wrapper as mujoco_playground_wrapper
from mujoco_playground._src import mjx_env

from ss2r.benchmark_suites import wrappers
from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def render(env, policy, steps, rng, camera=None):
    state = env.reset(rng)
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    videos = []
    for i in range(1):
        ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        video = env.render(ep_trajectory, camera=camera)
        videos.append(video)
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    vision: bool = False,
    num_vision_envs: int = 1,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
) -> mujoco_playground_wrapper.Wrapper:
    """Common wrapper pattern for all brax training agents.

    Args:
      env: environment to be wrapped
      vision: whether the environment will be vision based
      num_vision_envs: number of environments the renderer should generate,
        should equal the number of batched envs
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized model
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    if vision:
        env = mujoco_playground_wrapper.MadronaWrapper(
            env, num_vision_envs, randomization_fn
        )
    elif randomization_fn is None:
        env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
    else:
        env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = mujoco_playground_wrapper.BraxAutoResetWrapper(env)
    return env


class BraxDomainRandomizationVmapWrapper(wrappers.DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        (
            self._randomized_models,
            self._in_axes,
            self.domain_parameteres,
        ) = randomization_fn(self.mjx_model)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped._mjx_model = model
        return env
