import flax
import jax
import chex
import jax.numpy as jnp
from brax.training.types import Params

@flax.struct.dataclass
class ModelParams:
    """Contains training state for the model."""
    model: Params

def make_losses(model):
    """Create JIT-compiled loss functions for the model."""
    
    @jax.jit
    def _nll(
        predicted_outputs: chex.Array,
        predicted_stds: chex.Array,
        target_outputs: chex.Array,
        learn_std: bool
    ) -> chex.Array:
        if learn_std:
            log_prob = jax.scipy.stats.norm.logpdf(
                target_outputs, loc=predicted_outputs, scale=predicted_stds
            )
            return -jnp.mean(log_prob)
        else:
            loss = jnp.square(target_outputs - predicted_outputs).mean()
            return loss

    @jax.jit
    def _neg_log_posterior(
        predicted_outputs: chex.Array,
        predicted_stds: chex.Array,
        target_outputs: chex.Array,
        learn_std: bool
    ) -> chex.Array:
        # Expand target to match ensemble dimension
        target_expanded = jnp.expand_dims(target_outputs, axis=1)
        target_expanded = jnp.broadcast_to(target_expanded, predicted_outputs.shape)
        
        # Apply vmap over batch and ensemble dimensions
        nll = jax.vmap(jax.vmap(_nll, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None))(
            predicted_outputs, predicted_stds, target_expanded, learn_std
        )
        neg_log_post = nll.mean()
        return neg_log_post

    @jax.jit
    def compute_model_loss(
        model_params,
        normalizer_params,
        data,
        key,
        learn_std,
    ):
        model_apply = model.apply
        
        # Apply the model
        pred, std = model_apply(
            normalizer_params,
            model_params,
            key,
            data.observation,
            data.action
        )

        neg_log_prob = _neg_log_posterior(pred, std, data.next_observation, learn_std)
        
        # Broadcast target for MSE calculation
        target_expanded = jnp.expand_dims(data.next_observation, axis=1)
        target_expanded = jnp.broadcast_to(target_expanded, pred.shape)
        mse = jnp.mean(jnp.square(pred - target_expanded))

        return neg_log_prob, mse 
    
    return compute_model_loss