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
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(
                x
            )
            x = linen.LayerNorm()(x)
            x = self.activation(x)
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(
                x
            )
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
            # obs: [batch_size, obs_dim]
            # actions: [batch_size, action_dim]
            
            print(f"MModule input - obs: {obs.shape}, actions: {actions.shape}")
            
            hidden = jnp.concatenate([obs, actions], axis=-1)  # [batch_size, obs_dim + action_dim]
            
            res = []
            net = BroNet if use_bro else MLP
            output_dim = obs_size * 2 if predict_std else obs_size
            hidden_dims = list(hidden_layer_sizes) + [output_dim]

            for _ in range(self.n_ensemble):
                m = net(
                    hidden_dims=hidden_dims,
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)  # Each network outputs [batch_size, output_dim]
                res.append(m)
            
            # Stack along ensemble dimension: [batch_size, n_ensemble * output_dim]
            ensemble_output = jnp.concatenate(res, axis=-1)
            print(f"MModule output shape: {ensemble_output.shape}")
            
            return ensemble_output
        
    obs_size = _get_obs_state_size(obs_size, obs_key)
    model = MModule(n_ensemble=n_ensemble, obs_size=obs_size)

    def apply(processor_params, params, key, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        
        # Debug the input shapes
        print(f"Model input - obs: {obs.shape}, actions: {actions.shape}")
        
        # Get raw output from model
        raw_output = model.apply(params, obs, actions)
        print(f"Raw model output shape: {raw_output.shape}")
        print(f"predict_std: {predict_std}")
        
        batch_size = obs.shape[0]
        
        if predict_std:
            # Each ensemble member outputs both mean and std
            # raw_output shape: [batch_size, n_ensemble * obs_size * 2]
            expected_size = n_ensemble * obs_size * 2
            print(f"Expected size for predict_std=True: {expected_size}")
            
            # Reshape to [batch, ensemble, obs_dim * 2]
            reshaped = raw_output.reshape(batch_size, n_ensemble, obs_size * 2)
            
            # Split into mean and std
            pred_mean = reshaped[:, :, :obs_size]  # [batch, ensemble, obs_dim]
            pred_std = reshaped[:, :, obs_size:]   # [batch, ensemble, obs_dim]
            
            print(f"Reshaped - pred_mean: {pred_mean.shape}, pred_std: {pred_std.shape}")
            return pred_mean, pred_std
        else:
            # Each ensemble member outputs only mean
            # raw_output shape: [batch_size, n_ensemble * obs_size]
            expected_size = n_ensemble * obs_size
            print(f"Expected size for predict_std=False: {expected_size}")
            print(f"Actual raw_output size: {raw_output.shape[1]}")
            
            if raw_output.shape[1] != expected_size:
                raise ValueError(f"Output size mismatch: expected {expected_size}, got {raw_output.shape[1]}")
            
            # Reshape to [batch, ensemble, obs_dim]
            pred = raw_output.reshape(batch_size, n_ensemble, obs_size)
            std = jnp.ones_like(pred) * 1e-3
            
            print(f"Reshaped - pred: {pred.shape}, std: {std.shape}")
            return pred, std


    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), apply=apply
    )