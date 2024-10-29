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
    return lambda_ * jnp.std(state.obs, axis=0).mean()


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
        self.num_envs = None

    def reset(self, rng: jax.Array) -> State:
        if self.num_envs is None:
            self.num_envs = rng.shape[0]
        state = self.env.reset(rng)
        if "propagation_rng" in state.info:
            propagation_rng = state.info["propagation_rng"]
        else:
            propagation_rng = jax.random.split(rng[0])[1]
        n_key, key = jax.random.split(propagation_rng)
        state.info["propagation_rng"] = jax.random.split(n_key, self.num_envs)
        return self.propagation_fn(state, key)

    def step(self, state: State, action: jax.Array) -> State:
        # The order here matters, the tree_map changes the dimensions of
        # the propgattion_rng
        propagation_rng = state.info["propagation_rng"]
        tile = lambda tree: jax.tree_map(
            lambda x: jnp.tile(x, (self.num_envs,) + (1,) * x.ndim), tree
        )
        state, action = tile(state), tile(action)
        nstate = self.env.step(state, action)
        n_key, key = jax.random.split(propagation_rng)
        nstate.info["propagation_rng"] = jax.random.split(n_key, self.num_envs)
        if self.reward_bonus_fn is not None:
            nstate = nstate.replace(reward=nstate.reward + self.reward_bonus_fn(nstate))
        if self.cost_penalty_fn is not None:
            nstate.info["cost"] += self.cost_penalty_fn(nstate)
        return self.propagation_fn(nstate, key)


def get_randomized_values(sys_v, in_axes):
    sys_v_leaves, _ = jax.tree.flatten(sys_v)
    in_axes_leaves, _ = jax.tree.flatten(in_axes)
    randomized_values = [
        leaf for leaf, axis in zip(sys_v_leaves, in_axes_leaves) if axis is not None
    ]
    randomized_array = jnp.array(randomized_values).reshape(
        randomized_values[0].shape[0], len(randomized_values)
    )
    return randomized_array


class DomainRandomizationParams(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.domain_parameters = get_randomized_values(
            self.env._sys_v, self.env._in_axes
        )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["domain_parameters"] = self.domain_parameters
        return state
