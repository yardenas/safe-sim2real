from typing import Mapping

import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper
from brax.training.agents.sac import checkpoint

from ss2r.algorithms.sac.vision_networks import Encoder
from ss2r.common.wandb import get_wandb_checkpoint


class TrackOnlineCostsInObservation(Wrapper):
    def __init__(self, env):
        super().__init__(env)

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
            }
        return observation_size

    def augment_obs(self, state: State, cumulative_cost: jax.Array) -> State:
        if isinstance(state.obs, Mapping):
            obs = {
                **state.obs,
                "cumulative_cost": cumulative_cost,
            }
        else:
            obs = {
                "state": state.obs,
                "cumulative_cost": cumulative_cost,
            }
        return state.replace(obs=obs)

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        cummulative_cost = jnp.zeros_like(reset_state.reward, dtype=jnp.float32)[None]
        return self.augment_obs(reset_state, cummulative_cost)

    def step(self, state: State, action: jax.Array) -> State:
        cumulative_cost = state.obs["cumulative_cost"]
        old_obs = state.obs
        state = state.replace(obs=state.obs["state"])
        next_state = self.env.step(state, action)
        state = state.replace(obs=old_obs)
        cost = state.info.get("cost", jnp.zeros_like(state.reward, dtype=jnp.float32))
        next_cumulative_cost = cumulative_cost + cost
        next_state = self.augment_obs(next_state, next_cumulative_cost)
        return next_state


class VisionWrapper(Wrapper):
    def __init__(self, env, wandb_id, wandb_entity):
        super().__init__(env)
        checkpoint_path = get_wandb_checkpoint(wandb_id, wandb_entity)
        params = checkpoint.load(checkpoint_path)
        self.frozen_encoder_params = {"params": params[3]["params"]["SharedEncoder"]}
        self.encoder = Encoder()

    def reset(self, rng):
        state = super().reset(rng)
        return self._handle_state(state)

    def step(self, state, action):
        state = super().step(state, action)
        return self._handle_state(state)

    def _handle_state(self, state):
        assert isinstance(state.obs, Mapping)
        state.obs["latents"] = self.encoder.apply(self.frozen_encoder_params, state.obs)
        return state
