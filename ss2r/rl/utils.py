from functools import partial

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


def quantize_images(x):
    out = {}
    for k, v in x.items():
        if k.startswith("pixels/"):
            x = jnp.round(v * 255).astype(jnp.uint8)
            out[k] = x
    return {**x, **out}
