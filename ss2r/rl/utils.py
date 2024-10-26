import jax
from brax.envs import Env, State
from brax.training.types import Policy


def rollout(
    env: Env,
    policy: Policy,
    steps: int,
    rng: jax.Array,
    state: State | None = None,
) -> tuple[State, State]:
    parallel_envs = env._sys_v.m.shape
    keys = jax.random.split(rng, parallel_envs[0])
    if state is None:
        state = env.reset(keys)

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
