import jax
import jax.numpy as jnp
from brax.envs import State, Wrapper


class TrackOnlineCosts(Wrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.info["cumulative_cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        cumulative_cost = jnp.where(
            state.done,
            jnp.zeros_like(state.reward),
            state.info["cumulative_cost"],
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        nstate.info.update(cumulative_cost=cumulative_cost + cost)
        return nstate


class Saute(Wrapper):
    def __init__(
        self, env, episode_length, discounting, budget, penalty, terminate, lambda_
    ):
        super().__init__(env)
        # Assumes that this is the budget for the undiscounted
        # episode.
        self.budget = budget
        self.discounting = discounting
        self.terminate = terminate
        self.penalty = penalty
        self.disagreement_scale = lambda_

    @property
    def observation_size(self):
        observation_size = self.env.observation_size
        if isinstance(observation_size, dict):
            observation_size = {k: v + 1 for k, v in observation_size.items()}
        else:
            observation_size += 1
        return observation_size

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["saute_state"] = jnp.ones(())
        state.info["saute_reward"] = state.reward
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs=jnp.hstack([state.obs, state.info["saute_state"]])
            )
        else:
            obs = {
                k: jnp.hstack([v, state.info["saute_state"]])
                for k, v in state.obs.items()
            }
            state = state.replace(obs=obs)
        state.metrics["saute_reward"] = state.info["saute_reward"]
        state.metrics["saute_state"] = state.info["saute_state"]
        return state

    def step(self, state, action):
        saute_state = state.info["saute_state"]
        ones = jnp.ones_like(saute_state)
        saute_state = jnp.where(
            state.info.get("truncation", jnp.zeros_like(state.done)), ones, saute_state
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        cost += self.disagreement_scale * nstate.info.get("disagreement", 0.0)
        saute_state -= cost / self.budget
        saute_reward = jnp.where(saute_state <= 0.0, -self.penalty, nstate.reward)
        terminate = jnp.where(
            ((saute_state <= 0.0) & self.terminate) | nstate.done.astype(jnp.bool),
            True,
            False,
        )
        saute_state = jnp.where(terminate, ones, saute_state)
        nstate.info["saute_state"] = saute_state
        nstate.info["saute_reward"] = saute_reward
        nstate.metrics["saute_reward"] = saute_reward
        nstate.metrics["saute_state"] = saute_state
        if isinstance(nstate.obs, jax.Array):
            obs = jnp.hstack([nstate.obs, saute_state])
        else:
            obs = {k: jnp.hstack([v, saute_state]) for k, v in nstate.obs.items()}
        nstate = nstate.replace(
            obs=obs,
            done=terminate.astype(jnp.float32),
        )
        return nstate
