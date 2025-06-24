from typing import Callable, Mapping, Optional, Tuple

import jax
import jax.nn as jnn
from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training
from jax import numpy as jp
from mujoco_playground import State as MjxState


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
    hard_resets: bool = False,
    *,
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
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(
            env, randomization_fn, augment_state=augment_state
        )
    env = CostEpisodeWrapper(env, episode_length, action_repeat)
    if hard_resets:
        env = HardAutoResetWrapper(env)
    else:
        env = brax_training.AutoResetWrapper(env)
    return env


def _get_obs(state):
    if isinstance(state.obs, jax.Array):
        return state.obs
    else:
        assert isinstance(state.obs, Mapping)
        return state.obs["state"]


class SPiDR(Wrapper):
    def __init__(self, env, randomzation_fn, num_perturbed_envs, lambda_, alpha):
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
        self.lambda_ = lambda_
        self.alpha = alpha

    def reset(self, rng: jax.Array) -> State:
        # No need to randomize the initial state. Otherwise, even without
        # domain randomization, the initial states will be different, having
        # a non-zero disagreement.
        state = self.env.reset(rng)
        cost = jp.zeros_like(state.reward)
        state.info["state_propagation"] = {}
        state.info["state_propagation"]["next_obs"] = self._tile(_get_obs(state))
        state.info["state_propagation"]["cost"] = self._tile(cost)
        disagreement = self._compute_disagreement(
            state.info["state_propagation"]["next_obs"]
        )
        state.info["disagreement"] = disagreement
        state.metrics["disagreement"] = disagreement
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        v_state, v_action = self._tile(state), self._tile(action)
        perturbed_nstate = self.perturbed_env.step(v_state, v_action)
        next_obs = _get_obs(perturbed_nstate)
        disagreement = self._compute_disagreement(next_obs)
        nstate.info["state_propagation"]["next_obs"] = next_obs
        nstate.info["state_propagation"]["cost"] = perturbed_nstate.info.get(
            "cost", jp.zeros_like(perturbed_nstate.reward)
        )
        nstate.info["disagreement"] = disagreement
        nstate.metrics["disagreement"] = disagreement
        return nstate

    def _compute_disagreement(self, next_obs: jax.Array) -> jax.Array:
        variance = jp.nanvar(next_obs, axis=0).mean(-1)
        variance = jp.where(jp.isnan(variance), 0.0, variance)
        return jp.clip(variance, a_max=1000.0) * self.lambda_ + self.alpha

    def _tile(self, tree):
        def tile(x):
            x = jp.asarray(x)
            return jp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim)

        return jax.tree_map(tile, tree)


class BraxDomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.mjx_model)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped._mjx_model = model
        return env


class HardAutoResetWrapper(Wrapper):
    """Automatically reset Brax envs that are done.

    Resample only when >=1 environment is actually done. Still resamples for all
    """

    def reset(self, rng: jax.Array) -> State | MjxState:
        rng, sample_rng = jax.vmap(jax.random.split, out_axes=1)(rng)
        state = self.env.reset(sample_rng)
        state.info["reset_rng"] = rng
        return state

    def step(self, state: State | MjxState, action: jax.Array) -> State | MjxState:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        maybe_reset = jax.lax.cond(
            state.done.any(), self.reset, lambda rng: state, state.info["reset_rng"]
        )

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        if hasattr(state, "pipeline_state"):
            state_data = state.pipeline_state
            maybe_reset_data = maybe_reset.pipeline_state
            data_name = "pipeline_state"
        elif hasattr(state, "data"):
            state_data = state.data
            maybe_reset_data = maybe_reset.data
            data_name = "data"
        else:
            raise NotImplementedError
        new_data = jax.tree.map(where_done, maybe_reset_data, state_data)
        obs = jax.tree.map(where_done, maybe_reset.obs, state.obs)
        return state.replace(**{data_name: new_data, "obs": obs})


class Saute(Wrapper):
    def __init__(self, env, discounting, budget, penalty, terminate):
        super().__init__(env)
        # Assumes that this is the budget for the undiscounted
        # episode.
        self.budget = budget
        self.discounting = discounting
        self.terminate = terminate
        self.penalty = penalty

    @property
    def observation_size(self):
        observation_size = self.env.observation_size
        if isinstance(observation_size, dict):
            observation_size = {k: v + 1 for k, v in observation_size.items()}
        else:
            observation_size += 1
        return observation_size

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["saute_state"] = jp.ones(())
        state.info["eval_reward"] = state.reward
        if isinstance(state.obs, jax.Array):
            state = state.replace(obs=jp.hstack([state.obs, state.info["saute_state"]]))
        else:
            obs = {
                k: jp.hstack([v, state.info["saute_state"]])
                for k, v in state.obs.items()
            }
            state = state.replace(obs=obs)
        state.metrics["saute_unsafe"] = jp.zeros_like(state.reward)
        state.info["saute_reward"] = state.reward
        return state

    def step(self, state, action):
        saute_state = state.info["saute_state"]
        ones = jp.ones_like(saute_state)
        saute_state = jp.where(
            state.info.get("truncation", jp.zeros_like(state.done)), ones, saute_state
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jp.zeros_like(nstate.reward))
        cost += nstate.info.get("disagreement", 0.0)
        saute_state -= cost / self.budget
        # saute_reward = jp.where(saute_state <= 0.0, -self.penalty, nstate.reward)
        saute_reward = jnn.softplus(saute_state) * nstate.reward
        terminate = jp.where(
            ((saute_state <= 0.0) & self.terminate) | nstate.done.astype(jp.bool),
            True,
            False,
        )
        saute_state = jp.where(terminate, ones, saute_state)
        nstate.info["saute_state"] = saute_state
        nstate.info["eval_reward"] = nstate.reward
        nstate.info["saute_reward"] = saute_reward
        nstate.metrics["saute_unsafe"] = (saute_state <= 0.0).astype(jp.float32)
        if isinstance(nstate.obs, jax.Array):
            obs = jp.hstack([nstate.obs, saute_state])
        else:
            obs = {k: jp.hstack([v, saute_state]) for k, v in nstate.obs.items()}
        nstate = nstate.replace(
            obs=obs,
            done=terminate.astype(jp.float32),
            reward=saute_reward,
        )
        return nstate
