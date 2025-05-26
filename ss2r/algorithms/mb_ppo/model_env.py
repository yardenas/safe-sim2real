import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base
from brax.training.acme import running_statistics
from typing import Any, Dict

class ModelBasedEnv(base.Wrapper):
    """Environment wrapper that uses a learned model for predictions."""
    
    def __init__(
        self, 
        env: envs.Env,
        model_network,
        model_params,
        normalizer_params: running_statistics.RunningStatisticsState,
        ensemble_selection: str = "mean",  # "random", "mean", or "pessimistic"
        safety_budget: float = float("inf")
    ):
        super().__init__(env)
        self.model_network = model_network
        self.model_params = model_params
        self.normalizer_params = normalizer_params
        self.ensemble_selection = ensemble_selection
        self.safety_budget = safety_budget
        
    def reset(self, rng: jax.Array) -> base.State:
        """Reset using the real environment."""
        return self.env.reset(rng)
    
    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Get current observation
        obs = state.obs
        
        # Predict next state, reward, and cost using the model
        (next_obs_pred, reward_pred, cost_pred), (next_obs_std, reward_std, cost_std) = self.model_network.apply(
            self.normalizer_params,
            self.model_params,
            obs,
            action
        )
        
        # Select from ensemble
        if self.ensemble_selection == "random":
            key_ensemble_selection = jax.random.PRNGKey(jnp.sum(jnp.abs(obs))) # Simple way to get a somewhat varying key based on obs
            
            batch_size = next_obs_pred.shape[0]
            ensemble_size = next_obs_pred.shape[1]
            
            random_indices = jax.random.randint(key_ensemble_selection, (batch_size,), 0, ensemble_size)
            
            next_obs = jax.vmap(lambda arr, idx: arr[idx])(next_obs_pred, random_indices)
            reward = jax.vmap(lambda arr, idx: arr[idx])(reward_pred, random_indices)
            cost = jax.vmap(lambda arr, idx: arr[idx])(cost_pred, random_indices)

        elif self.ensemble_selection == "mean":
            # Use ensemble mean
            next_obs = jnp.mean(next_obs_pred, axis=1)
            reward = jnp.mean(reward_pred, axis=1)
            cost = jnp.mean(cost_pred, axis=1)
        elif self.ensemble_selection == "pessimistic":
            next_obs = jnp.mean(next_obs_pred, axis=1) # Or a more sophisticated pessimistic obs
            reward = jnp.min(reward_pred, axis=1)  # Pessimistic reward
            cost = jnp.max(cost_pred, axis=1)      # Pessimistic cost
        else:
            raise ValueError(f"Unknown ensemble selection: {self.ensemble_selection}")
        
        reward = reward.squeeze(axis=-1) if reward.ndim == next_obs.ndim else reward
        cost = cost.squeeze(axis=-1) if cost.ndim == next_obs.ndim else cost

        done = jnp.zeros_like(reward,dtype=jnp.float32)

        prev_cumulative_cost = state.info.get("cumulative_cost", jnp.zeros_like(cost))
        accumulated_cost_for_transition = prev_cumulative_cost + cost

        if self.safety_budget < float("inf"):
            done = jnp.where(accumulated_cost_for_transition > self.safety_budget, jnp.ones_like(done), done)
        accumulated_cost_for_transition = jnp.where(done > 0, jnp.zeros_like(accumulated_cost_for_transition), accumulated_cost_for_transition)
        
        model_extras = {
            "cost": cost, # cost should be (batch_size,) 
            "cumulative_cost": accumulated_cost_for_transition, # (batch_size,)
            "truncation": jnp.zeros_like(done,dtype=jnp.float32),  # (batch_size,)
        }
        
        new_info = {**state.info, **model_extras}
        
        new_state = base.State(
            pipeline_state=state.pipeline_state,
            obs=next_obs,
            reward=reward, # reward should be (batch_size,)
            done=done,
            metrics=state.metrics, # Preserve metrics
            info=new_info
        )
        
        return new_state


def create_model_env(
    env: envs.Env,
    model_network,
    model_params,
    normalizer_params: running_statistics.RunningStatisticsState,
    ensemble_selection: str = "random",
    safety_budget: float = float("inf"),
) -> ModelBasedEnv:
    """Factory function to create a model-based environment."""
    return ModelBasedEnv(
        env=env,
        model_network=model_network,
        model_params=model_params,
        normalizer_params=normalizer_params,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget
    )