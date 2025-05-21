from typing import Mapping, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper

from ss2r.benchmark_suites.mujoco_playground import BraxDomainRandomizationVmapWrapper
from ss2r.benchmark_suites.wrappers import DomainRandomizationVmapWrapper


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
        def tile(x):
            x = jnp.asarray(x)
            return jnp.tile(x, (self.num_perturbed_envs,) + (1,) * x.ndim)

        return jax.tree_map(tile, tree)


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
        variance = jnp.clip(variance, a_max=1000.0)
        nstate.info["disagreement"] = variance
        nstate.metrics["disagreement"] = variance
        return nstate
