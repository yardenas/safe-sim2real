from typing import Mapping, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper

from ss2r.benchmark_suites.mujoco_playground import BraxDomainRandomizationVmapWrapper


class PropagationFn(Protocol):
    def __call__(self, state: State, rng: jax.Array) -> State:
        ...


def ts1(state, rng):
    num_envs = _get_obs(state).shape[0]
    id_ = jax.random.randint(rng, (), 0, num_envs)
    sampled_state = jax.tree_map(lambda x: x[id_], state, is_leaf=eqx.is_array)
    return sampled_state


def _get_obs(state):
    if isinstance(state.obs, jax.Array):
        return state.obs
    else:
        assert isinstance(state.obs, Mapping)
        return state.obs["state"]


class PTSD(Wrapper):
    def __init__(self, env, randomzation_fn, num_perturbed_envs):
        super().__init__(env)
        self.perturbed_env = BraxDomainRandomizationVmapWrapper(
            env, randomzation_fn, augment_state=False
        )
        self.num_perturbed_envs = num_perturbed_envs

    def reset(self, rng: jax.Array) -> State:
        # No need to randomize the initial state. Otherwise, even without
        # domain randomization, the initial states will be different, having
        # a non-zero disagreement.
        state = self.env.reset(rng)
        cost = jnp.zeros_like(state.reward)
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
            "cost", jnp.zeros_like(perturbed_nstate.reward)
        )
        return nstate

    def _tile(self, tree):
        return jax.tree_map(
            lambda x: jnp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim), tree
        )


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
        orig_next_obs = _get_obs(state)
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
        n_key, key = jax.random.split(propagation_rng)
        orig_next_obs = _get_obs(nstate)
        nstate.info["state_propagation"]["rng"] = jax.random.split(n_key, self.num_envs)
        nstate.info["state_propagation"]["next_obs"] = nstate.obs
        nstate = self.propagation_fn(nstate, key)
        nstate.info["state_propagation"]["next_obs"] = orig_next_obs
        return nstate


class ModelDisagreement(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jnp.nanvar(next_obs, axis=0).mean(-1)
        variance = jnp.where(jnp.isnan(variance), 0.0, variance)
        state.info["disagreement"] = variance
        state.metrics["disagreement"] = variance
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        next_obs = state.info["state_propagation"]["next_obs"]
        variance = jnp.nanvar(next_obs, axis=0).mean(-1)
        variance = jnp.where(jnp.isnan(variance), 0.0, variance)
        nstate.info["disagreement"] = variance
        nstate.metrics["disagreement"] = variance
        return nstate
