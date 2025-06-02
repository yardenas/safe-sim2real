import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base
from brax.training.acme import running_statistics


class ModelBasedEnv(envs.Env):
    """Environment wrapper that uses a learned model for predictions."""

    def __init__(
        self,
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

    def reset(self, rng: jax.Array) -> base.State:
        """Reset using the real environment."""
        raise NotImplementedError(
            "ModelBasedEnv does not support reset. Use a real environment for resetting."
        )

    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Predict next state, reward, and cost using the model
        (
            (diff_next_obs_pred, reward_pred, cost_pred),
            (diff_next_obs_std, reward_std, cost_std),
        ) = self.model_network.apply(
            self.normalizer_params, self.model_params, state.obs, action
        )
        # Select from ensemble
        # TODO (manu): refactor this to a separate function (self._propagate_method)
        if self.ensemble_selection == "random":
            key_ensemble_selection = jax.random.PRNGKey(jnp.sum(jnp.abs(state.obs)))
            batch_size = diff_next_obs_pred.shape[0]
            ensemble_size = diff_next_obs_pred.shape[1]
            random_indices = jax.random.randint(
                key_ensemble_selection, (batch_size,), 0, ensemble_size
            )
            next_obs = state.obs + jax.vmap(lambda arr, idx: arr[idx])(
                diff_next_obs_pred, random_indices
            )
            reward = jax.vmap(lambda arr, idx: arr[idx])(reward_pred, random_indices)
            cost = jax.vmap(lambda arr, idx: arr[idx])(cost_pred, random_indices)
        elif self.ensemble_selection == "mean":
            # Use ensemble mean
            next_obs = state.obs + jnp.mean(diff_next_obs_pred, axis=0)
            reward = jnp.mean(reward_pred, axis=0)
            cost = jnp.mean(cost_pred, axis=0)
        elif self.ensemble_selection == "pessimistic":
            next_obs = state.obs + jnp.mean(diff_next_obs_pred, axis=0)
            reward = jnp.min(reward_pred, axis=0)
            cost = jnp.max(cost_pred, axis=0)
        else:
            raise ValueError(f"Unknown ensemble selection: {self.ensemble_selection}")
        done = jnp.zeros_like(reward, dtype=jnp.float32)
        prev_cumulative_cost = state.info["cumulative_cost"]
        accumulated_cost_for_transition = prev_cumulative_cost + cost
        if self.safety_budget < float("inf"):
            done = jnp.where(
                accumulated_cost_for_transition > self.safety_budget,
                jnp.ones_like(done),
                done,
            )
        accumulated_cost_for_transition = jnp.where(
            done > 0,
            jnp.zeros_like(accumulated_cost_for_transition),
            accumulated_cost_for_transition,
        )
        state.info["cumulative_cost"] = accumulated_cost_for_transition
        state.info["cost"] = cost
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
        model_network=model_network,
        observation_size=observation_size,
        action_size=action_size,
        model_params=model_params,
        normalizer_params=normalizer_params,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget,
    )
