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
        # Log initial shapes
        print(f"INPUT - state.reward shape: {state.reward.shape if hasattr(state.reward, 'shape') else 'scalar'}")
        print(f"INPUT - state.info['cost'] shape: {state.info['cost'].shape if 'cost' in state.info and hasattr(state.info['cost'], 'shape') else 'not available'}")
        print(f"INPUT - state.obs shape: {state.obs.shape if hasattr(state.obs, 'shape') else 'scalar'}")
        print(f"INPUT - action shape: {action.shape if hasattr(action, 'shape') else 'scalar'}")
        
        # Get current observation
        obs = state.obs
        
        # Remember original reward shape for exact matching
        input_reward_shape = state.reward.shape if hasattr(state.reward, 'shape') else None
        print(f"Captured input_reward_shape: {input_reward_shape}")
        
        # Remember original observation properties
        original_shape = obs.shape
        original_dtype = obs.dtype
        was_1d = len(original_shape) == 1
        
        # Ensure observations have the right shape for the model network
        if not hasattr(obs, 'ndim') or not hasattr(obs, 'shape'):
            obs = jnp.asarray(obs)
    
        # Ensure obs is at least 2D (batch, features)
        if obs.ndim == 1:  # If obs is just (features,)
            obs = obs[None, :]  # Add batch dimension -> (1, features)
    
        # Ensure action has the right shape
        if not hasattr(action, 'ndim') or not hasattr(action, 'shape'):
            action = jnp.asarray(action)
        if action.ndim == 1:  # If action is just (features,)
            action = action[None, :]  # Add batch dimension -> (1, features)
    
        # Log shapes after initial processing
        print(f"PROCESSED - obs shape: {obs.shape}")
        print(f"PROCESSED - action shape: {action.shape}")
    
        # Predict next state, reward, and cost using the model
        (next_obs_pred, reward_pred, cost_pred), (next_obs_std, reward_std, cost_std) = self.model_network.apply(
            self.normalizer_params,
            self.model_params,
            obs,
            action
        )
        
        # Log model output shapes
        print(f"MODEL OUTPUT - next_obs_pred shape: {next_obs_pred.shape}")
        print(f"MODEL OUTPUT - reward_pred shape: {reward_pred.shape}")
        print(f"MODEL OUTPUT - cost_pred shape: {cost_pred.shape}")
    
        # Select from ensemble
        if self.ensemble_selection == "random":
            # Log shapes before random selection
            print(f"BEFORE RANDOM - next_obs_pred: {next_obs_pred.shape}")
            
            key_ensemble_selection = jax.random.PRNGKey(jnp.sum(jnp.abs(obs)))
            
            batch_size = next_obs_pred.shape[0]
            ensemble_size = next_obs_pred.shape[1]
            
            random_indices = jax.random.randint(key_ensemble_selection, (batch_size,), 0, ensemble_size)
            
            next_obs = jax.vmap(lambda arr, idx: arr[idx])(next_obs_pred, random_indices)
            reward = jax.vmap(lambda arr, idx: arr[idx])(reward_pred, random_indices)
            cost = jax.vmap(lambda arr, idx: arr[idx])(cost_pred, random_indices)
            
            # Log shapes after random selection
            print(f"AFTER RANDOM - next_obs: {next_obs.shape}, reward: {reward.shape}, cost: {cost.shape}")

        elif self.ensemble_selection == "mean":
            # Use ensemble mean
            next_obs = jnp.mean(next_obs_pred, axis=1)
            reward = jnp.mean(reward_pred, axis=1)
            cost = jnp.mean(cost_pred, axis=1)
            
            # Log shapes after mean
            print(f"AFTER MEAN - next_obs: {next_obs.shape}, reward: {reward.shape}, cost: {cost.shape}")
            
        elif self.ensemble_selection == "pessimistic":
            next_obs = jnp.mean(next_obs_pred, axis=1)
            reward = jnp.min(reward_pred, axis=1)
            cost = jnp.max(cost_pred, axis=1)
            
            # Log shapes
            print(f"AFTER PESSIMISTIC - next_obs: {next_obs.shape}, reward: {reward.shape}, cost: {cost.shape}")
        else:
            raise ValueError(f"Unknown ensemble selection: {self.ensemble_selection}")
    
        # Restore original shape and dtype of observations
        if was_1d:
            next_obs = next_obs.squeeze(axis=0)
            print(f"AFTER SQUEEZE - next_obs shape: {next_obs.shape}")
    
        # Ensure the dtype matches
        next_obs = next_obs.astype(original_dtype)
    
        # Before squeezing
        print(f"BEFORE SQUEEZE - reward shape: {reward.shape}, cost shape: {cost.shape}")
    
        reward = reward.squeeze(axis=-1) if reward.ndim == next_obs.ndim + 1 else reward
        cost = cost.squeeze(axis=-1) if cost.ndim == next_obs.ndim + 1 else cost
    
        # After squeezing
        print(f"AFTER SQUEEZE - reward shape: {reward.shape}, cost shape: {cost.shape}")
    
        # FIX: Check if we need to broadcast reward and cost to match input shape
        if input_reward_shape is not None and input_reward_shape != reward.shape:
            if reward.size == 1:  # If we have a scalar or single element
                print(f"BROADCASTING reward from {reward.shape} to {input_reward_shape}")
                reward = jnp.broadcast_to(reward.reshape((1,)), input_reward_shape)
            # Handle the (512, 1) to (512,) case
            elif reward.shape[0] == input_reward_shape[0] and reward.ndim > len(input_reward_shape):
                print(f"RESHAPING reward from {reward.shape} to {input_reward_shape}")
                reward = jnp.reshape(reward, input_reward_shape)
            else:
                print(f"WARNING: Cannot reshape reward from {reward.shape} to {input_reward_shape}")

        if input_reward_shape is not None and 'cost' in state.info and input_reward_shape != cost.shape:
            if cost.size == 1:  # If we have a scalar or single element
                print(f"BROADCASTING cost from {cost.shape} to {input_reward_shape}")
                cost = jnp.broadcast_to(cost.reshape((1,)), input_reward_shape) 
            # Handle the (512, 1) to (512,) case
            elif cost.shape[0] == input_reward_shape[0] and cost.ndim > len(input_reward_shape):
                print(f"RESHAPING cost from {cost.shape} to {input_reward_shape}")
                cost = jnp.reshape(cost, input_reward_shape)
            else:
                print(f"WARNING: Cannot reshape cost from {cost.shape} to {input_reward_shape}")

        done = jnp.zeros_like(reward, dtype=jnp.float32)
        print(f"done shape: {done.shape}")

        prev_cumulative_cost = state.info.get("cumulative_cost", jnp.zeros_like(cost))
        print(f"prev_cumulative_cost shape: {prev_cumulative_cost.shape}")
        
        accumulated_cost_for_transition = prev_cumulative_cost + cost
        print(f"accumulated_cost_for_transition shape: {accumulated_cost_for_transition.shape}")

        if self.safety_budget < float("inf"):
            done = jnp.where(accumulated_cost_for_transition > self.safety_budget, jnp.ones_like(done), done)
        accumulated_cost_for_transition = jnp.where(done > 0, jnp.zeros_like(accumulated_cost_for_transition), accumulated_cost_for_transition)
        
        model_extras = {
            "cost": cost, # cost should be (batch_size,) 
            "cumulative_cost": accumulated_cost_for_transition, # (batch_size,)
            "truncation": jnp.zeros_like(done, dtype=jnp.float32),  # (batch_size,)
        }
        
        new_info = {**state.info, **model_extras}
        
        # Log final shapes before returning
        print(f"FINAL - next_obs: {next_obs.shape}, reward: {reward.shape}, done: {done.shape}")
        
        new_state = base.State(
            pipeline_state=state.pipeline_state,
            obs=next_obs,
            reward=reward, 
            done=done,
            metrics=state.metrics, 
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