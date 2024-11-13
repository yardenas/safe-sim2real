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


class LCB(QTransformation):
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_

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
        next_obs = transitions.extras["state_extras"]["state_propagation"]["next_obs"]
        next_action, _ = policy(next_obs)
        if domain_params is not None:
            domain_params = jnp.tile(
                domain_params[:, None], (1, next_action.shape[1], 1)
            )
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(next_obs, next_action)
        next_v = next_q.mean(axis=-1)
        std = jnp.std(next_v, axis=-1)
        cost = transitions.extras["state_extras"]["cost"]
        cost += self.lambda_ * std
        target_q = jax.lax.stop_gradient(
            cost * reward_scaling + transitions.discount * gamma * next_v
        )
        return target_q


class CVaR(QTransformation):
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

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
        next_obs = transitions.extras["state_extras"]["state_propagation"]["next_obs"]
        next_action, _ = policy(next_obs)
        if domain_params is not None:
            domain_params = jnp.tile(
                domain_params[:, None], (1, next_action.shape[1], 1)
            )
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(next_obs, next_action)
        next_v = next_q.mean(axis=-1)
        sort_next_v = jnp.sort(next_v, axis=-1)
        cvar_index = int((1 - self.confidence) * next_v.shape[1])
        next_v = jnp.mean(sort_next_v[:, :cvar_index], axis=-1)
        cost = transitions.extras["state_extras"]["cost"]
        target_q = jax.lax.stop_gradient(
            cost * reward_scaling + transitions.discount * gamma * next_v
        )
        return target_q


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
        cost = transitions.extras["state_extras"]["cost"]
        target_q = jax.lax.stop_gradient(cost + transitions.discount * gamma * next_v)
        return target_q
