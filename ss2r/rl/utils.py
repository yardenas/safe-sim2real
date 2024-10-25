import jax
from brax.training.types import Policy, State


def rollout(
    self,
    policy: Policy,
    steps: int,
    rng: int,
    state: State | None = None,
) -> tuple[State, State]:
    if state is None:
        rng, rng = jax.random.split(rng)
        keys = jax.random.split(rng, self.parallel_envs)
        state = self.environment.reset(keys)

    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = self.step(
            state,
            policy,
            current_key,
            extra_fields=("truncation",),
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(f, (state, rng), (), length=steps)
    return final_state, data
