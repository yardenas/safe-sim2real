from typing import Mapping

import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


class TrackOnlineCostsInObservation(Wrapper):
    def __init__(self, env, cost_discount=1.0):
        super().__init__(env)
        self.cost_discount = cost_discount

    @property
    def observation_size(self):
        observation_size = self.env.observation_size
        if isinstance(observation_size, dict):
            observation_size = {k: v for k, v in observation_size.items()}
            observation_size.update(cumulative_cost=1)
        else:
            observation_size = {
                "state": observation_size,
                "cumulative_cost": 1,
                "curr_discount": 1,
            }
        return observation_size

    def augment_obs(self, state: State, cumulative_cost, curr_discount) -> State:
        if isinstance(state.obs, Mapping):
            obs = {
                **state.obs,
                "cumulative_cost": cumulative_cost,
                "curr_discount": curr_discount,
            }
        else:
            # Append the cumulative cost to state.obs
            # obs = jnp.concatenate([state.obs, cumulative_cost], axis=-1)
            obs = {
                "state": state.obs,
                "cumulative_cost": cumulative_cost,
                "curr_discount": curr_discount,
            }
        return state.replace(obs=obs)

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        cummulative_cost = jnp.zeros_like(reset_state.reward)[None]
        curr_discount = jnp.ones_like(reset_state.reward)[None]
        return self.augment_obs(reset_state, cummulative_cost, curr_discount)

    def step(self, state: State, action: jax.Array) -> State:
        cumulative_cost = state.obs["cumulative_cost"]
        curr_discount = state.obs["curr_discount"] * self.cost_discount
        old_obs = state.obs
        state = state.replace(obs=state.obs["state"])
        next_state = self.env.step(state, action)
        state = state.replace(obs=old_obs)
        cost = state.info.get("cost", jnp.zeros_like(state.reward))
        next_cumulative_cost = cumulative_cost + cost * curr_discount
        next_state = self.augment_obs(next_state, next_cumulative_cost, curr_discount)
        return next_state
