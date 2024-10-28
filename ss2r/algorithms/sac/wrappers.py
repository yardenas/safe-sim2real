from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


class PropagationFn(Protocol):
    def __call__(self, state: State, rng: jax.Array) -> State:
        ...


def ts1(state, rng):
    num_envs = state.obs.shape[0]
    id_ = jax.random.randint(rng, (), 0, num_envs)
    sampled_state = jax.tree_map(lambda x: x[id_], state, is_leaf=eqx.is_array)
    return sampled_state


def std_bonus(state, lambda_):
    return lambda_ * jnp.std(state.obs, axis=1)


class StatePropagation(Wrapper):
    """
    Wrapper for adding action and observation delays in Brax envs, using JAX.
    This wrapper assumes that the environment is wrapped before with a VmapWrapper or DomainRandomizationVmapWrapper
    """

    def __init__(
        self, env, propagation_fn=ts1, reward_bonus_fn=None, cost_penalty_fn=None
    ):
        super().__init__(env)
        self.reward_bonus_fn = reward_bonus_fn
        self.cost_penalty_fn = cost_penalty_fn
        self.propagation_fn = propagation_fn

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_obs"] = state.obs
        if "propagation_rng" in state.info:
            ts_rng = state.info["propagation_rng"]
        else:
            ts_rng = jax.random.split(rng)[1]
        n_key, key = jax.random.split(ts_rng)
        state.info["propagation_rng"] = n_key
        return self.propagation_fn(state, key)

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        ts_rng = state.info["propagation_rng"]
        n_key, key = jax.random.split(ts_rng)
        state.info["propagation_rng"] = n_key
        if self.reward_bonus_fn is not None:
            nstate.reward += self.reward_bonus_fn(nstate)
        if self.cost_penalty_fn is not None:
            nstate.info["cost"] += self.cost_penalty_fn(nstate)
        return self.propagation_fn(nstate, key)
