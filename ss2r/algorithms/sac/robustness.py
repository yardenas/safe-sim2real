from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from brax.training.types import Params, Transition
from jax.scipy.stats import norm


class QTransformation(Protocol):
    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
    ) -> jax.Array:
        ...


class UCBCost(QTransformation):
    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
    ):
        next_action, _ = policy(transitions.next_observation)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(transitions.next_observation, next_action)
        next_v = next_q.mean(axis=-1)
        std = transitions.extras["state_extras"]["disagreement"]
        cost = transitions.extras["state_extras"]["cost"] + self.lambda_ * std
        target_q = jax.lax.stop_gradient(cost + transitions.discount * gamma * next_v)
        return target_q


class RAMU(QTransformation):
    """
    https://arxiv.org/pdf/2301.12593
    """

    def __init__(self, epsilon: float, n_samples: int, wang_eta: float) -> None:
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.wang_eta = wang_eta

    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
    ):
        next_action, _ = policy(transitions.next_observation)
        sampled_next_obs = self.sample(
            transitions.observation, transitions.next_observation
        )
        if domain_params is not None:
            domain_params = jnp.tile(
                domain_params[:, None], (1, next_action.shape[1], 1)
            )
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(sampled_next_obs, next_action)
        next_v = next_q.mean(axis=-1)
        quantiles = jnp.linspace(0, 1, self.n_samples + 1)
        quantiles = norm.cdf(norm.ppf(quantiles) + self.wang_eta)
        probs = (quantiles[1:] - quantiles[:-1]) * self.n_samples
        next_v = (jnp.sort(next_v, axis=-1) * probs).mean(-1)
        cost = transitions.extras["state_extras"]["cost"]
        target_q = jax.lax.stop_gradient(cost + transitions.discount * gamma * next_v)
        return target_q

    def sample(self, observation, next_observation):
        delta = next_observation - observation
        x = jax.random.uniform(
            0,
            (self.n_samples, *delta.shape),
            minval=-2.0 * self.epsilon,
            maxval=2.0 * self.epsilon,
        )
        return observation + (delta) * (1.0 + x)


class SACBase(QTransformation):
    def __init__(self, alpha: float, reward_scaling: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.reward_scaling = reward_scaling

    def __call__(
        self,
        transitions: Transition,
        q_fn: Callable[[Params, jax.Array], jax.Array],
        policy: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        gamma: float,
        domain_params: jax.Array | None = None,
    ):
        next_action, next_log_prob = policy(transitions.next_observation)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(transitions.next_observation, next_action)
        next_v = next_q.min(axis=-1)
        next_v -= self.alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * self.reward_scaling
            + transitions.discount * gamma * next_v
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
    ):
        next_action, _ = policy(transitions.next_observation)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_fn(transitions.next_observation, next_action)
        next_v = next_q.mean(axis=-1)
        cost = transitions.extras["state_extras"]["cost"]
        target_q = jax.lax.stop_gradient(cost + transitions.discount * gamma * next_v)
        return target_q
