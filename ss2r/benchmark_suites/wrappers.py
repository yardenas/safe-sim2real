from typing import Callable, Optional, Tuple

import jax
from brax.base import System
from brax.envs import wrappers
from brax.envs.base import Env, State, Wrapper
from jax import numpy as jp


class ActionObservationDelayWrapper(Wrapper):
    """Wrapper for adding action and observation delays in Brax envs, using JAX."""

    def __init__(self, env, action_delay: int = 0, obs_delay: int = 0):
        super().__init__(env)
        self.action_delay = action_delay
        self.obs_delay = obs_delay

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # Initialize the action and observation buffers as part of the state.info
        action_buffer, obs_buffer = self._init_buffers(state)
        # Store buffers in the state info for later access
        state.info["action_buffer"] = action_buffer
        state.info["obs_buffer"] = obs_buffer
        return state

    def _init_buffers(self, state):
        # Initialize the action and observation buffers as part of the state.info
        zero_action = jp.zeros(self.env.action_size)
        action_buffer = jp.tile(zero_action[None], (self.action_delay + 1, 1))
        obs_buffer = jp.tile(state.obs[None], (self.obs_delay + 1, 1))
        # Store buffers in the state info for later access
        return action_buffer, obs_buffer

    def step(self, state: State, action: jax.Array) -> State:
        # Retrieve the buffers from the state info
        action_buffer = state.info["action_buffer"]
        obs_buffer = state.info["obs_buffer"]
        # Shift the buffers to add new action and observation (delayed behavior)
        new_action_buffer = jp.roll(action_buffer, shift=-1, axis=0)
        new_action_buffer = new_action_buffer.at[-1].set(action)
        delayed_action = new_action_buffer[0]
        # Step the environment using the delayed action
        state = self.env.step(state, delayed_action)
        # Shift the observation buffer and add the current observation
        new_obs_buffer = jp.roll(obs_buffer, shift=-1, axis=0)
        new_obs_buffer = new_obs_buffer.at[-1].set(state.obs)
        delayed_obs = new_obs_buffer[0]

        # Update state observation with the delayed observation
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        init_action, init_obs = self._init_buffers(state)
        new_obs_buffer = where_done(init_obs, new_obs_buffer)
        new_action_buffer = where_done(init_action, new_action_buffer)
        # Update the buffers in state.info and return the updated state
        state.info["action_buffer"] = new_action_buffer
        state.info["obs_buffer"] = new_obs_buffer
        state = state.replace(
            obs=delayed_obs,
        )
        return state


class FrameActionStack(Wrapper):
    """Wrapper that stacks both observations and actions in a rolling manner for Brax environments.

    This wrapper maintains a history of both observations and actions, allowing the agent to access
    temporal information. For the initial state, the observation buffer is filled with the initial
    observation, and the action buffer is filled with zeros.

    Args:
        env: The Brax environment to wrap
        num_stack: Number of frames to stack (applies to both observations and actions)
    """

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack

        # Modify observation space to account for stacked frames and actions
        # Note: In Brax, we don't explicitly define spaces like in Gymnasium
        # but we'll track the dimensions for clarity
        self.single_obs_shape = self.env.observation_size
        self.single_action_shape = self.env.action_size
        self.num_stack = num_stack

    @property
    def observation_size(self) -> int:
        return self.num_stack * (self.single_obs_shape + self.single_action_shape)

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment and initialize the frame and action stacks."""
        state = self.env.reset(rng)
        # Create initial observation stack (filled with initial observation)
        action_buffer, obs_buffer = self._init_buffers(state)
        state.info["action_stack"] = action_buffer
        state.info["obs_stack"] = obs_buffer
        # Create the stacked observation
        state = state.replace(
            obs=self._get_stacked_obs(obs_buffer, action_buffer),
        )
        return state

    def _init_buffers(self, state):
        # Initialize the action and observation buffers as part of the state.info
        zero_action = jp.zeros(self.single_action_shape)
        action_buffer = jp.tile(zero_action[None], (self.num_stack, 1))
        obs_buffer = jp.tile(state.obs[None], (self.num_stack, 1))
        # Store buffers in the state info for later access
        return action_buffer, obs_buffer

    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment and update the stacks."""
        # Get current stacks
        action_buffer = state.info["action_stack"]
        obs_buffer = state.info["obs_stack"]
        # Step the environment
        state = self.env.step(state, action)
        # Update observation stack
        new_obs_buffer = jp.roll(obs_buffer, shift=-1, axis=0)
        new_obs_buffer = new_obs_buffer.at[-1].set(state.obs)

        # Update action stack
        new_action_buffer = jp.roll(action_buffer, shift=-1, axis=0)
        new_action_buffer = new_action_buffer.at[-1].set(action)

        # Handle done states
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        # Create the stacked observation
        stacked_obs = self._get_stacked_obs(new_obs_buffer, new_action_buffer)
        init_action, init_obs = self._init_buffers(state)
        new_obs_buffer = where_done(init_obs, new_obs_buffer)
        new_action_buffer = where_done(init_action, new_action_buffer)
        # Update state
        state.info["action_stack"] = new_action_buffer
        state.info["obs_stack"] = new_obs_buffer
        state = state.replace(
            obs=stacked_obs,
        )
        return state

    def _get_stacked_obs(
        self, obs_stack: jax.Array, action_stack: jax.Array
    ) -> jax.Array:
        """Combine the observation and action stacks into a single observation."""
        # Flatten the observation stack
        flat_obs = obs_stack.reshape(-1)
        # Flatten the action stack
        flat_actions = action_stack.reshape(-1)
        # Concatenate them
        return jp.concatenate([flat_obs, flat_actions])


class DomainRandomizationVmapWrapper(Wrapper):
    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[System], Tuple[System, System, jax.Array]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes, self.domain_parameters = randomization_fn(self.sys)
        dummy = self.env.reset(jax.random.PRNGKey(0))
        self.strip_privileged_state = isinstance(dummy.obs, jax.Array)

    def _env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            state = env.reset(rng)
            return state

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        state = self._add_privileged_state(state)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def step(sys, s, a):
            env = self._env_fn(sys=sys)
            state = env.step(s, a)
            return state

        if self.strip_privileged_state:
            in_state = state.replace(obs=state.obs["state"])
        else:
            in_state = state
        state = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
            self._sys_v, in_state, action
        )
        state = self._add_privileged_state(state)
        return state

    def _add_privileged_state(self, state: State) -> State:
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs={
                    "state": state.obs,
                    "privileged_state": jp.concatenate(
                        [state.obs, self.domain_parameters], -1
                    ),
                }
            )
        else:
            state = state.replace(
                obs={
                    "state": state.obs["state"],
                    "privileged_state": jp.concatenate(
                        [state.obs["privileged_state"], self.domain_parameters], -1
                    ),
                }
            )
        return state

    @property
    def observation_size(self) -> dict[str, int]:
        if isinstance(self.env.observation_size, int):
            return {
                "state": self.env.observation_size,
                "privileged_state": self.env.observation_size
                + self.domain_parameters.shape[1],
            }
        else:
            return {
                "state": self.env.observation_size["state"],
                "privileged_state": self.env.observation_size["privileged_state"]
                + self.domain_parameters.shape[1],
            }


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System, jax.Array]]
    ] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = wrappers.training.EpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = wrappers.training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = wrappers.training.AutoResetWrapper(env)
    return env
