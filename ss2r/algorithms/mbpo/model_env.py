import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base
from brax.training.acme import running_statistics

from ss2r.algorithms.sac.types import float32


class ModelBasedEnv(envs.Env):
    """Environment wrapper that uses a learned model for predictions."""

    def __init__(
        self,
        transitions,
        observation_size: int,
        action_size: int,
        model_network,
        model_params,
        normalizer_params: running_statistics.RunningStatisticsState,
        ensemble_selection: str = "mean",  # "random", "mean", or "pessimistic"
        safety_budget: float = float("inf"),
    ):
        super().__init__()
        self.model_network = model_network
        self.model_params = model_params
        self.normalizer_params = normalizer_params
        self.ensemble_selection = ensemble_selection
        self.safety_budget = safety_budget
        self._observation_size = observation_size
        self._action_size = action_size
        self.transitions = transitions

    def reset(self, rng: jax.Array) -> base.State:
        sample_key, model_key = jax.random.split(rng)
        indcs = jax.random.randint(
            sample_key, (), 0, self.transitions.observation.shape[0]
        )
        transitions = float32(
            jax.tree_util.tree_map(lambda x: x[indcs], self.transitions)
        )
        state = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=jnp.zeros_like(transitions.reward),
            info={
                "cumulative_cost": transitions.extras["state_extras"].get(
                    "cumulative_cost", jnp.zeros_like(transitions.reward)
                ),
                "curr_discount": transitions.extras["state_extras"].get(
                    "curr_discount", jnp.ones_like(transitions.reward)
                ),
                "truncation": jnp.zeros_like(transitions.reward),
                "cost": transitions.extras["state_extras"].get(
                    "cost", jnp.zeros_like(transitions.reward)
                ),
                "key": model_key,
            },
        )
        return state

    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Predict next state, reward, and cost using the model
        key = state.info["key"]
        sample_key, key = jax.random.split(key)
        next_obs, reward, cost = _propagate_ensemble(
            self.model_network.apply,
            self.normalizer_params,
            self.model_params,
            state.obs,
            action,
            self.ensemble_selection,
            sample_key,
        )
        state.info["key"] = key
        done = jnp.zeros_like(reward, dtype=jnp.float32)
        truncation = jnp.zeros_like(reward, dtype=jnp.float32)
        state.info["cost"] = cost
        state.info["truncation"] = truncation
        if "cumulative_cost" in state.info:
            prev_cumulative_cost = state.info["cumulative_cost"]
            curr_discount = state.info.get("curr_discount", jnp.ones_like(reward))
            accumulated_cost_for_transition = (
                prev_cumulative_cost + curr_discount * cost
            )
            done = jnp.where(
                accumulated_cost_for_transition > self.safety_budget,
                jnp.ones_like(done),
                done,
            )

            def reset_states(self, done, state, next_obs):
                """Reset the state if done."""
                key, reset_keys = jax.random.split(state.info["key"])
                state.info["key"] = key
                state.info["cumulative_cost"] = jnp.where(
                    done,
                    jnp.zeros_like(reward),
                    state.info["cumulative_cost"],
                )
                next_obs = jnp.where(done, self.reset(reset_keys).obs, next_obs)
                return state, next_obs

            state, next_obs = reset_states(self, done, state, next_obs)

        state = state.replace(
            obs=next_obs,
            reward=reward,
            done=done,
            info=state.info,
        )
        return state

    def observation_size(self) -> int:
        """Return the size of the observation space."""
        return self._observation_size

    def action_size(self) -> int:
        """Return the size of the action space."""
        return self._action_size

    def backend(self) -> str:
        """Return the backend used by the environment."""
        return "model_based"


def create_model_env(
    transitions,
    model_network,
    model_params,
    observation_size: int,
    action_size: int,
    normalizer_params: running_statistics.RunningStatisticsState,
    ensemble_selection: str = "random",
    safety_budget: float = float("inf"),
) -> ModelBasedEnv:
    """Factory function to create a model-based environment."""
    return ModelBasedEnv(
        transitions=transitions,
        model_network=model_network,
        observation_size=observation_size,
        action_size=action_size,
        model_params=model_params,
        normalizer_params=normalizer_params,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget,
    )


def _propagate_ensemble(
    pred_fn,
    normalizer_params,
    model_params,
    obs,
    action,
    ensemble_selection,
    key,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate the ensemble predictions based on the selection method."""
    # Calculate the nominal predictions
    if ensemble_selection == "nominal":
        # Get the average model parameters
        avg_model_params = jax.tree_util.tree_map(
            lambda p: jnp.mean(p, axis=0), model_params
        )
        next_obs, reward, cost = pred_fn(
            normalizer_params, avg_model_params, obs, action
        )
    elif ensemble_selection == "random":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        # Randomly select one of the ensemble predictions
        idx = jax.random.randint(key, (1,), 0, next_obs_pred.shape[0])[0]
        next_obs = next_obs_pred[idx]
        reward = reward_pred[idx]
        cost = cost_pred[idx]
    elif ensemble_selection == "mean":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        next_obs = jnp.mean(next_obs_pred, axis=0)
        reward = jnp.mean(reward_pred, axis=0)
        cost = jnp.mean(cost_pred, axis=0)
    else:
        raise ValueError(f"Unknown ensemble selection: {ensemble_selection}")
    return next_obs, reward, cost
