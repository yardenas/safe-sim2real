from functools import partial
from typing import Mapping

import jax
import jax.numpy as jnp
from brax.envs import Env, State
from brax.training.types import Policy


@partial(jax.jit, static_argnames=("env", "policy", "steps"))
def rollout(
    env: Env,
    policy: Policy,
    steps: int,
    rng: jax.Array,
    state: State,
) -> tuple[State, State]:
    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        action, _ = policy(state.obs, current_key)
        nstate = env.step(
            state,
            action,
        )
        return (nstate, next_key), nstate

    (final_state, _), data = jax.lax.scan(f, (state, rng), (), length=steps)
    return final_state, data


def quantize_images(observations):
    if not isinstance(observations, Mapping):
        return observations
    out = {}
    for k, v in observations.items():
        if k.startswith("pixels/"):
            observations = jnp.round(v * 255).astype(jnp.uint8)
            out[k] = observations
        else:
            out[k] = v
    return out


def dequantize_images(observations):
    if not isinstance(observations, Mapping):
        return observations
    out = {}
    for k, v in observations.items():
        if k.startswith("pixels/"):
            observations = v.astype(jnp.float32) / 255
            out[k] = observations
        else:
            out[k] = v
    return out
