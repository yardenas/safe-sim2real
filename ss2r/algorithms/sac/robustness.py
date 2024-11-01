from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from brax.training.types import Params, Transition


class QTransformation(Protocol):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
        alpha: jax.Array | None = None,
        reward_scaling: float = 1.0,
    ):
        ...


# class CVaR(QTransformation):
#     def __call__(
#         self,
#         transitions: Transition,
#         q_fn: Callable[[Params, jax.Array], jax.Array],
#         policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
#         gamma: float,
#         domain_params: jax.Array | None = None,
#         alpha: jax.Array | None = None,
#         reward_scaling: float = 1.0,
#     ):
#         return super().__call__(transitions, q_fn)


class SACBase(QTransformation):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
        alpha: jax.Array | None = None,
        reward_scaling: float = 1.0,
    ):
        next_action, next_log_prob = policy(transitions.next_observation)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(transitions.next_observation, next_action)
        next_v = next_q.min(axis=-1)
        next_v -= alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * gamma * next_v
        )
        return target_q


class SACCost(QTransformation):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
        alpha: jax.Array | None = None,
        reward_scaling: float = 1.0,
    ):
        next_action, _ = policy(transitions.next_observation)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(transitions.next_observation, next_action)
        next_v = next_q.mean(axis=-1)
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * gamma * next_v
        )
        return target_q
