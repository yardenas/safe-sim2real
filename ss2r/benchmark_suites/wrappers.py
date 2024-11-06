import jax
import jax.numpy as jp
from brax.envs import State, Wrapper


class ActionObservationDelayWrapper(Wrapper):
    """Wrapper for adding action and observation delays in Brax envs, using JAX."""

    def __init__(self, env, action_delay: int = 0, obs_delay: int = 0):
        super().__init__(env)
        self.action_delay = action_delay
        self.obs_delay = obs_delay

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # Initialize the action and observation buffers as part of the state.info
        zero_action = jp.zeros_like(state.obs)
        action_buffer = jp.tile(
            zero_action[None], (self.action_delay + 1,) + zero_action.shape
        )
        obs_buffer = jp.tile(state.obs[None], (self.obs_delay + 1,) + state.obs.shape)
        # Store buffers in the state info for later access
        state.info["action_buffer"] = action_buffer
        state.info["obs_buffer"] = obs_buffer
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state

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

        pipeline_state = jax.tree_map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = where_done(state.info["first_obs"], delayed_obs)
        # Update the buffers in state.info and return the updated state
        state = state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            info={
                "action_buffer": new_action_buffer,
                "obs_buffer": new_obs_buffer,
                "first_pipeline_state": state.info["first_pipeline_state"],
                "first_obs": state.info["first_obs"],
            },
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
        return self.single_obs_shape * self.num_stack + self.single_action_shape * (
            self.num_stack - 1
        )

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment and initialize the frame and action stacks."""
        state = self.env.reset(rng)

        # Create initial observation stack (filled with initial observation)
        obs_stack = jp.tile(state.obs[None], (self.num_stack,) + state.obs.shape)

        # Create initial action stack (filled with zeros)
        zero_action = jp.zeros(self.single_action_shape)
        action_stack = jp.tile(
            zero_action[None], (self.num_stack - 1,) + zero_action.shape
        )

        # Store stacks in state info
        state.info["obs_stack"] = obs_stack
        state.info["action_stack"] = action_stack

        # Create the stacked observation
        state = state.replace(
            obs=self._get_stacked_obs(obs_stack, action_stack),
            info={
                **state.info,
                "obs_stack": obs_stack,
                "action_stack": action_stack,
                "first_obs": state.obs,
            },
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Step the environment and update the stacks."""
        # Get current stacks
        obs_stack = state.info["obs_stack"]
        action_stack = state.info["action_stack"]

        # Step the environment
        state = self.env.step(state, action)

        # Update observation stack
        new_obs_stack = jp.roll(obs_stack, shift=-1, axis=0)
        new_obs_stack = new_obs_stack.at[-1].set(state.obs)

        # Update action stack
        new_action_stack = jp.roll(action_stack, shift=-1, axis=0)
        new_action_stack = new_action_stack.at[-1].set(action)

        # Handle done states
        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        # Create the stacked observation
        stacked_obs = self._get_stacked_obs(new_obs_stack, new_action_stack)
        obs = where_done(state.info["first_obs"], stacked_obs)

        # Update state
        state = state.replace(
            obs=obs,
            info={
                **state.info,
                "obs_stack": new_obs_stack,
                "action_stack": new_action_stack,
            },
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
