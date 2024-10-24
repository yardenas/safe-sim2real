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
