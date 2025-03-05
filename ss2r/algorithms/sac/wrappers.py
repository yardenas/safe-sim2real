from typing import Mapping, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


class PropagationFn(Protocol):
    def __call__(self, state: State, rng: jax.Array) -> State:
        ...


def ts1(state, rng):
    num_envs = _get_state(state).shape[0]
    id_ = jax.random.randint(rng, (), 0, num_envs)
    sampled_state = jax.tree_map(lambda x: x[id_], state, is_leaf=eqx.is_array)
    return sampled_state


def _get_state(state):
    if isinstance(state.obs, jax.Array):
        return state.obs
    else:
        assert isinstance(state.obs, Mapping)
        return state.obs["state"]


class StatePropagation(Wrapper):
    """
    Wrapper for adding action and observation delays in Brax envs, using JAX.
    This wrapper assumes that the environment is wrapped before with a VmapWrapper or DomainRandomizationVmapWrapper
    """

    def __init__(self, env, propagation_fn=ts1):
        super().__init__(env)
        self.propagation_fn = propagation_fn
        self.num_envs = None

    def reset(self, rng: jax.Array) -> State:
        # TODO (yarden): this code is not jax compatible.
        if self.num_envs is None:
            self.num_envs = rng.shape[0]
        # No need to randomize the initial state. Otherwise, even without
        # domain randomization, the initial states will be different, having
        # a non-zero disagreement.
        rng = jnp.tile(rng[:1], (self.num_envs, 1))
        state = self.env.reset(rng)
        propagation_rng = jax.random.split(rng[0])[1]
        n_key, key = jax.random.split(propagation_rng)
        state.info["state_propagation"] = {}
        state.info["state_propagation"]["rng"] = jax.random.split(n_key, self.num_envs)
        orig_next_obs = _get_state(state)
        state = self.propagation_fn(state, key)
        state.info["state_propagation"]["next_obs"] = orig_next_obs
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # The order here matters, the tree_map changes the dimensions of
        # the propgattion_rng
        propagation_rng = state.info["state_propagation"]["rng"]
        tile = lambda tree: jax.tree_map(
            lambda x: jnp.tile(x, (self.num_envs,) + (1,) * x.ndim), tree
        )
        state, action = tile(state), tile(action)
        nstate = self.env.step(state, action)
        tmp_n_key, key = jax.random.split(propagation_rng)
        n_key = jnp.where(
            nstate.done.astype(jnp.bool) | state.info["truncation"].astype(jnp.bool),
            tmp_n_key,
            propagation_rng,
        )
        orig_next_obs = _get_state(nstate)
        nstate.info["state_propagation"]["rng"] = jax.random.split(n_key, self.num_envs)
        nstate.info["state_propagation"]["next_obs"] = nstate.obs
        nstate = self.propagation_fn(nstate, key)
        nstate.info["state_propagation"]["next_obs"] = orig_next_obs
        return nstate


class ModelDisagreement(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.env, StatePropagation)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jnp.var(next_obs, axis=1).mean(-1)
        state.info["disagreement"] = variance
        state.metrics["disagreement"] = variance
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jnp.var(next_obs, axis=1).mean(-1)
        nstate.info["disagreement"] = variance
        nstate.metrics["disagreement"] = variance
        return nstate
