from typing import Callable, Mapping, Optional, Tuple

import jax
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training
from jax import numpy as jp

from ss2r.benchmark_suites.mujoco_playground import BraxDomainRandomizationVmapWrapper


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


class DomainRandomizationVmapBase(Wrapper):
    """Base class for domain randomization wrappers."""

    def __init__(self, env, randomization_fn, *, augment_state=True):
        super().__init__(env)
        self.augment_state = augment_state
        (
            self._randomized_models,
            self._in_axes,
            self.domain_parameters,
        ) = self._init_randomization(randomization_fn)
        dummy = self.env.reset(jax.random.PRNGKey(0))
        self.strip_privileged_state = isinstance(dummy.obs, jax.Array)

    def _init_randomization(self, randomization_fn):
        """To be implemented by subclasses to handle model-specific randomization."""
        raise NotImplementedError

    def _env_fn(self, model):
        """To be implemented by subclasses to return an environment with the given model."""
        raise NotImplementedError

    def reset(self, rng: jax.Array):
        def reset_fn(model, rng):
            env = self._env_fn(model)
            return env.reset(rng)

        state = jax.vmap(reset_fn, in_axes=[self._in_axes, 0])(
            self._randomized_models, rng
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def step(self, state, action: jax.Array):
        def step_fn(model, s, a):
            env = self._env_fn(model)
            return env.step(s, a)

        if self.augment_state and self.strip_privileged_state:
            state = state.replace(obs=state.obs["state"])

        state = jax.vmap(step_fn, in_axes=[self._in_axes, 0, 0])(
            self._randomized_models, state, action
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def _add_privileged_state(self, state):
        """Adds privileged state to the observation if augmentation is enabled."""
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
    def observation_size(self):
        """Compute observation size based on the augmentation setting."""
        if not self.augment_state:
            return self.env.observation_size

        if isinstance(self.env.observation_size, int):
            return {
                "state": (self.env.observation_size,),
                "privileged_state": (
                    self.env.observation_size + self.domain_parameters.shape[1],
                ),
            }
        else:
            return {
                "state": (self.env.observation_size["state"],),
                "privileged_state": (
                    self.env.observation_size["privileged_state"]
                    + self.domain_parameters.shape[1],
                ),
            }


class DomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.sys)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped.sys = model
        return env


class CostEpisodeWrapper(brax_training.EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            maybe_cost = nstate.info.get("cost", None)
            return nstate, (nstate.reward, maybe_cost)

        state, (rewards, maybe_costs) = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        if maybe_costs is not None:
            state.info["cost"] = jp.sum(maybe_costs, axis=0)
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System, jax.Array]]
    ] = None,
    augment_state: bool = True,
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
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(
            env, randomization_fn, augment_state=augment_state
        )
    env = brax_training.AutoResetWrapper(env)
    return env


def _get_obs(state):
    if isinstance(state.obs, jax.Array):
        return state.obs
    else:
        assert isinstance(state.obs, Mapping)
        return state.obs["state"]


class SPiDR(Wrapper):
    def __init__(self, env, randomzation_fn, num_perturbed_envs):
        super().__init__(env)
        if hasattr(env, "sys"):
            self.perturbed_env = DomainRandomizationVmapWrapper(
                env, randomzation_fn, augment_state=False
            )
        elif hasattr(env, "mjx_model"):
            self.perturbed_env = BraxDomainRandomizationVmapWrapper(
                env, randomzation_fn, augment_state=False
            )
        else:
            raise ValueError("Should be either mujoco playground or brax env")
        self.num_perturbed_envs = num_perturbed_envs

    def reset(self, rng: jax.Array) -> State:
        # No need to randomize the initial state. Otherwise, even without
        # domain randomization, the initial states will be different, having
        # a non-zero disagreement.
        state = self.env.reset(rng)
        cost = jp.zeros_like(state.reward)
        state.info["state_propagation"] = {}
        state.info["state_propagation"]["next_obs"] = self._tile(_get_obs(state))
        state.info["state_propagation"]["cost"] = self._tile(cost)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        v_state, v_action = self._tile(state), self._tile(action)
        perturbed_nstate = self.perturbed_env.step(v_state, v_action)
        next_obs = _get_obs(perturbed_nstate)
        nstate.info["state_propagation"]["next_obs"] = next_obs
        nstate.info["state_propagation"]["cost"] = perturbed_nstate.info.get(
            "cost", jp.zeros_like(perturbed_nstate.reward)
        )
        return nstate

    def _tile(self, tree):
        def tile(x):
            x = jp.asarray(x)
            return jp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim)

        return jax.tree_map(tile, tree)


class ModelDisagreement(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jp.nanvar(next_obs, axis=0).mean(-1)
        variance = jp.where(jp.isnan(variance), 0.0, variance)
        state.info["disagreement"] = variance
        state.metrics["disagreement"] = variance
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jp.nanvar(next_obs, axis=0).mean(-1)
        variance = jp.where(jp.isnan(variance), 0.0, variance)
        variance = jp.clip(variance, a_max=1000.0)
        nstate.info["disagreement"] = variance
        nstate.metrics["disagreement"] = variance
        return nstate
