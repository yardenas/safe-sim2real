import flax
import jax
import chex
import jax.numpy as jnp
from brax.training.types import Params

@flax.struct.dataclass
class ModelParams:
    """Contains training state for the model."""

    model: Params

def make_losses(
        model
):
    def compute_model_loss(
        model_params,
        normalizer_params,
        data,
        key,
        learn_std,
    ):
        
        def _neg_log_posterior(
            predicted_outputs: chex.Array,  # Shape: (batch, ensemble, obs_dim)
            predicted_stds: chex.Array,     # Shape: (batch, ensemble, obs_dim)
            target_outputs: chex.Array      # Shape: (batch, obs_dim)
        ) -> chex.Array:
            # Expand target to match ensemble dimension
            target_expanded = jnp.expand_dims(target_outputs, axis=1)  # (batch, 1, obs_dim)
            target_expanded = jnp.broadcast_to(target_expanded, predicted_outputs.shape)  # (batch, ensemble, obs_dim)
            
            print(f"Loss shapes - pred: {predicted_outputs.shape}, target: {target_expanded.shape}")
            
            # Apply vmap over batch and ensemble dimensions
            nll = jax.vmap(jax.vmap(_nll, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))(
                predicted_outputs, predicted_stds, target_expanded
            )
            neg_log_post = nll.mean()
            return neg_log_post

        def _nll(
            predicted_outputs: chex.Array,
            predicted_stds: chex.Array,
            target_outputs: chex.Array
        ) -> chex.Array:
            if learn_std:
                log_prob = jax.scipy.stats.norm.logpdf(
                    target_outputs, loc=predicted_outputs, scale=predicted_stds
                )
                return -jnp.mean(log_prob)
            else:
                loss = jnp.square(target_outputs - predicted_outputs).mean()
                return loss

        model_apply = model.apply
        
        # Fixed: Add the missing key parameter and correct argument order
        pred, std = model_apply(
            normalizer_params,     # processor_params
            model_params,          # params
            key,                   # key (was missing!)
            data.observation,      # obs
            data.action           # actions
        )

        neg_log_prob = _neg_log_posterior(pred, std, data.next_observation)
        
        # Broadcast target for MSE calculation
        target_expanded = jnp.expand_dims(data.next_observation, axis=1)  # (batch, 1, obs_dim)
        target_expanded = jnp.broadcast_to(target_expanded, pred.shape)   # (batch, ensemble, obs_dim)
        mse = jnp.mean(jnp.square(pred - target_expanded))

        return neg_log_prob, mse 
    
    return compute_model_loss