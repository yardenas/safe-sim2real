from functools import partial
from typing import Mapping, Union

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
            quantized = jnp.round(v * 255).astype(jnp.uint8)
            out[k] = quantized
        else:
            out[k] = v
    return out


def dequantize_images(observations):
    if not isinstance(observations, Mapping):
        return observations
    out = {}
    for k, v in observations.items():
        if k.startswith("pixels/"):
            unquantized = v.astype(jnp.float32) / 255
            out[k] = unquantized
        else:
            out[k] = v
    return out


def remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def restore_state(tree, target_example):
    state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(target_example), jax.tree_util.tree_leaves(tree)
    )
    return state
