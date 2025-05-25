from typing import Sequence, Callable

import jax
import jax.numpy as jnp
from brax.training.networks import ActivationFn, FeedForwardNetwork, MLP
from flax import linen
from brax.training import types
from brax.training.networks import _get_obs_state_size

class BroNet(linen.Module):
    hidden_dims: Sequence[int]
    activation: Callable
    kernel_init: Callable = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        assert all(size == self.hidden_dims[0] for size in self.hidden_dims[:-1])
        x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
        x = linen.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(len(self.hidden_dims) - 1):
            residual = x
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
            x = linen.LayerNorm()(x)
            x = self.activation(x)
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
            x = linen.LayerNorm()(x)
            x += residual
        x = linen.Dense(self.hidden_dims[-1], kernel_init=self.kernel_init)(x)
        return x

def make_world_model_ensemble(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_ensemble: int = 5,
    obs_key: str = "state",
    use_bro: bool = True,
    predict_std: bool = False,
) -> FeedForwardNetwork:
    """Creates a model network."""

    class MModule(linen.Module):
        """M Module."""
        n_ensemble: int
        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            # Remove print statements for JIT compatibility
            hidden = jnp.concatenate([obs, actions], axis=-1)
            
            res = []
            net = BroNet if use_bro else MLP
            output_dim = obs_size * 2 if predict_std else obs_size
            hidden_dims = list(hidden_layer_sizes) + [output_dim]

            for _ in range(self.n_ensemble):
                m = net(
                    hidden_dims=hidden_dims,
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(m)
            
            ensemble_output = jnp.concatenate(res, axis=-1)
            return ensemble_output
        
    obs_size = _get_obs_state_size(obs_size, obs_key)
    model = MModule(n_ensemble=n_ensemble, obs_size=obs_size)

    @jax.jit
    def apply(processor_params, params, key, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        
        # Get raw output from model
        raw_output = model.apply(params, obs, actions)
        
        batch_size = obs.shape[0]
        
        if predict_std:
            expected_size = n_ensemble * obs_size * 2
            reshaped = raw_output.reshape(batch_size, n_ensemble, obs_size * 2)
            pred_mean = reshaped[:, :, :obs_size]
            pred_std = reshaped[:, :, obs_size:]
            return pred_mean, pred_std
        else:
            expected_size = n_ensemble * obs_size
            pred = raw_output.reshape(batch_size, n_ensemble, obs_size)
            std = jnp.ones_like(pred) * 1e-3
            return pred, std

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), 
        apply=apply
    )