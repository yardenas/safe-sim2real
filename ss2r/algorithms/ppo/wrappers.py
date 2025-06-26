import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


class TrackOnlineCosts(Wrapper):
    def __init__(self, env, cost_discount=1.0):
        super().__init__(env)
        self.cost_discount = cost_discount

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.info["cumulative_cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        reset_state.info["curr_discount"] = jnp.ones_like(reset_state.reward)
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        cumulative_cost = jnp.where(
            state.done,
            jnp.zeros_like(state.reward),
            state.info["cumulative_cost"],
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        curr_discount = nstate.info.get("curr_discount", jnp.ones_like(nstate.reward))
        nstate.info.update(cumulative_cost=cumulative_cost + curr_discount * cost)
        nstate.info.update(curr_discount=curr_discount * self.cost_discount)
        return nstate
