from typing import Callable, TypedDict

import jax
import jax.numpy as jnp

ObjectiveFn = Callable[[jax.Array], jax.Array]


def make_objective(
    rollout_fn,
    horizon: int,
    initial_state: jax.Array,
    key: jax.Array,
) -> ObjectiveFn:
    def objective(candidates):
        sample = lambda x: rollout_fn(horizon, initial_state, key, x)
        preds = jax.vmap(sample)(candidates)
        assert preds.reward.ndim == 2
        return preds.reward.mean(axis=1)

    return objective


def solve(
    objective_fn: ObjectiveFn,
    initial_guess: jax.Array,
    key: jax.Array,
    num_particles: int,
    num_iters: int,
    num_elite: int,
    stop_cond: float = 0.1,
    initial_stddev: float = 1.0,
) -> jax.Array:
    mu = initial_guess
    stddev = jnp.ones_like(initial_guess) * initial_stddev

    def cond(val):
        _, iters, _, stddev, *_ = val
        return (stddev.mean() > stop_cond) & (iters < num_iters)

    def body(val):
        key, iter, mu, stddev, best_score, best = val
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, (num_particles,) + mu.shape)
        sample = eps * stddev[None] + mu[None]
        scores = objective_fn(sample)
        elite_ids = jnp.argsort(scores)[-num_elite:]
        best = jnp.where(
            scores[elite_ids[-1]] > best_score, sample[elite_ids[-1]], best
        )
        best_score = jnp.maximum(best_score, scores[elite_ids[-1]])
        elite = sample[elite_ids]
        # Moment matching on the `particles` axis
        mu, stddev = elite.mean(0), elite.std(0)
        return key, iter + 1, mu, stddev, best_score, best

    *_, best = jax.lax.while_loop(
        cond, body, (key, 0, mu, stddev, -jnp.inf, initial_guess)
    )
    return best


class CEMConfig(TypedDict):
    num_particles: int
    num_iters: int
    num_elite: int
    stop_cond: float
    initial_stddev: float


def policy(
    observation: jax.Array,
    rollout_fn,
    horizon: int,
    init_guess: jax.Array,
    key: jax.Array,
    cem_config: CEMConfig,
) -> jax.Array:
    objective = make_objective(rollout_fn, horizon, observation, key)
    action = solve(
        objective,
        init_guess,
        key,
        **cem_config,
    )
    return action
