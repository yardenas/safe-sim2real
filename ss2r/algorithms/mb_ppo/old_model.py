import functools

import chex
import jax.random
import flax.linen as nn
import flax
from typing import Dict, Callable, Sequence
import jax.numpy as jnp
import optax
import os
from jaxtyping import PyTree
from jax.scipy.stats import norm
from brax.training.types import identity_observation_preprocessor


from ss2r.algorithms.mb_ppo.networks import make_world_model_ensemble


@chex.dataclass
class EnsembleState:
    vmapped_params: PyTree
    step: int

class DeterministicEnsemble(object):
    def __init__(self,
                 observation_size: int,
                 action_size: int,
                 hidden_layer_sizes: Sequence[int] = (256, 256),
                 num_heads: int = 5,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
                 use_bro: bool = True,
                 obs_key: str = "state",
                 preprocess_observations_fn = identity_observation_preprocessor
                 ):
        
        self.num_heads = num_heads
        self.obs_size = observation_size
        self.action_size = action_size
        self.step = 0
        self.params = None
        
        self.model = make_world_model_ensemble(
            obs_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            n_ensemble=num_heads,
            obs_key=obs_key,
            use_bro=use_bro
        )
        self.learn_std = False


    def init(self, key: jax.random.PRNGKey, input: jnp.ndarray):
        obs_size = self.obs_size
        obs = input[:, :obs_size]
        actions = input[:, obs_size:]
        
        model_params = self.model.init(key)
        
        out = self.model.apply(None, model_params, obs, actions)
        out = out.reshape(-1, self.num_heads, obs_size)

        self.step = 0
        self.params = model_params


    def __call__(self, input):
        out_mean, out_std = self.apply(input, params=self.params)
        return out_mean, out_std

    def apply(self, input, params):
        obs_size = self.obs_size
        obs = input[:, :obs_size]
        actions = input[:, obs_size:]
        
        outputs = self.model.apply(None, params, obs, actions)
        outputs = outputs.reshape(-1, self.num_heads, obs_size)
        outputs = jnp.transpose(outputs, (1, 0, 2))
        
        return outputs, jnp.ones_like(outputs) * 1e-3

    def _neg_log_posterior(self,
                           predicted_outputs: chex.Array,
                           predicted_stds: chex.Array,
                           target_outputs: chex.Array) -> chex.Array:
        nll = jax.vmap(jax.vmap(self._nll), in_axes=(0, 0, None))(predicted_outputs, predicted_stds, target_outputs)
        neg_log_post = nll.mean()
        return neg_log_post

    def _nll(self,
             predicted_outputs: chex.Array,
             predicted_stds: chex.Array,
             target_outputs: chex.Array) -> chex.Array:
        if self.learn_std:
            log_prob = norm.logpdf(target_outputs, loc=predicted_outputs, scale=predicted_stds)
            return -jnp.mean(log_prob)
        else:
            loss = jnp.square(target_outputs - predicted_outputs).mean()
            return loss

    def loss(self, params, input, y):
        out, std = self.apply(input, params=params)
        neg_log_prob = self._neg_log_posterior(predicted_outputs=out, predicted_stds=std, target_outputs=y)
        mse = jnp.square(out - y[jnp.newaxis]).mean()
        return neg_log_prob, mse

    def update(self, input, output):
        params, opt_state, step = self.params, opt_state, self.step
        (loss, mse), grads = jax.value_and_grad(self.loss, has_aux=True)(params, input, output)
        updates, new_opt_state = self.tx.update(grads, opt_state,
                                                params)
        new_params = optax.apply_updates(params, updates)

        self.params = new_params
        self.step = step + 1

        return (loss, mse)

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def get_info_gain(self, input, state: EnsembleState):
        mean, std = self(input=input, state=state, denormalize_output=False)
        al_std = jnp.clip(jnp.sqrt(jnp.square(std).mean(0)), 1e-3, None)
        ep_std = mean.std(axis=0)
        ratio = jnp.square(ep_std / al_std)
        info_gain = jnp.log(1 + ratio).mean(axis=-1).reshape(-1, 1)
        return info_gain.reshape(-1)


class ProbabilisticEnsemble(DeterministicEnsemble):
    def __init__(self, model_kwargs: Dict, sig_min: float = 1e-3, sig_max: float = 1e2, *args, **kwargs):
        hidden_dims = list(model_kwargs['hidden_dims'])
        hidden_dims[-1] = 2 * hidden_dims[-1]
        hidden_dims = tuple(hidden_dims)
        model_kwargs['hidden_dims'] = hidden_dims
        self.sig_min = sig_min
        self.sig_max = sig_max
        super().__init__(model_kwargs=model_kwargs, *args, **kwargs)
        self.learn_std = True

    def apply_single(self, input, params):
        out = self.model.apply(params, input)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig

    def apply(self, input, params):
        return jax.vmap(self.apply_single, in_axes=(None, 0))(input, params)


def main():
    key = jax.random.PRNGKey(0)
    state_dim = 2         # 2D state [s₁, s₂]
    action_dim = 1        # 1D action [a]
    data_size = 1000      # Number of training samples
    num_heads = 5         # Number of ensemble models
    batch_size = 32       # Batch size for training
    
    # Generate random states and actions
    key, state_key, action_key = jax.random.split(key, 3)
    states = jax.random.uniform(state_key, (data_size, state_dim), minval=-3.0, maxval=3.0)
    actions = states[:,0].reshape(-1, 1) * jnp.cos(states[:,0]).reshape(-1, 1)     
    # Generate next states using the dynamics: [s₁+a, s₂-a]
    next_states = jnp.zeros_like(states)
    next_states = next_states.at[:, 0].set(
        states[:, 0] * jnp.cos(states[:, 0]) + 0.3 * actions[:, 1] + 0.1 * states[:, 1]**2
    )
    next_states = next_states.at[:, 1].set(
        states[:, 1] * jnp.sin(states[:, 1]) - 0.5 * actions[:, 0] + 0.2 * states[:, 0] * states[:, 1]
    )
    
    # Add noise to make the task more realistic
    noise_level = 0.05
    key, noise_key = jax.random.split(key)
    next_states += noise_level * jax.random.normal(noise_key, next_states.shape)
    
    # Initialize the optimizer
    tx = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    
    # Initialize the ensemble model
    ensemble = DeterministicEnsemble(
        observation_size=state_dim,   # State dimension
        action_size=action_dim,       # Action dimension
        hidden_layer_sizes=[64, 64],  # Hidden layer sizes
        optimizer=tx,
        num_heads=num_heads)
    
    dummy_s = states[:1]  # One sample state [1, state_dim]
    dummy_a = actions[:1] # One sample action [1, action_dim]
    ensemble_state = ensemble.init(key=key, input=jnp.concatenate([dummy_s, dummy_a], axis=-1))
    
    # Training loop
    num_steps = 5000
    training_indices = []
    for i in range(num_steps):
        # Sample a batch of data
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(batch_key, (batch_size,), 0, data_size)
        training_indices.append(idx)
        s_batch = states[idx]
        a_batch = actions[idx]
        s_next = next_states[idx]
        
        # Update the model
        ensemble_state, (loss, mse) = ensemble.update(
            input=jnp.concatenate([s_batch, a_batch], axis=-1),  # Concatenate state and action
            output=s_next,   # Next states are outputs
            state=ensemble_state
        )
        
        # Print progress
        if i % 200 == 0:
            info_gain = ensemble.get_info_gain(
                input=a_batch, 
            )
            print(f'Step: {i}, Loss: {loss:.4f}, MSE: {mse:.4f}, Info Gain: {info_gain.mean():.4f}')
    
    # Evaluation
    # Generate a grid of states and a fixed action for visualization
    test_s1 = jnp.linspace(-3.0, 3.0, 20).reshape(-1,)
    test_s2 = jnp.linspace(-3.0, 3.0, 20).reshape(-1,)
    test_s1, test_s2 = jnp.meshgrid(test_s1, test_s2)
    test_states = jnp.stack([test_s1.flatten(), test_s2.flatten()], axis=1)
    
    # Actions as sinousoidal function of linspace
    test_actions = test_states[:,0].reshape(-1, 1) * jnp.cos(test_states[:,0]).reshape(-1, 1)     
    # Get ensemble predictions
    preds, std = ensemble(jnp.concatenate([test_states, test_actions], axis=-1), ensemble_state)
    preds_mean = preds.mean(axis=0)
    ep_std = preds.std(axis=0)
    
    # True next states for comparison
    true_next = jnp.zeros_like(test_states)
    true_next = true_next.at[:, 0].set(
        test_states[:, 0] * jnp.cos(test_states[:, 0]) + 0.3 * test_actions[:, 1] + 0.1 * test_states[:, 1]**2
    )
    true_next = true_next.at[:, 1].set(
        test_states[:, 1] * jnp.sin(test_states[:, 1]) - 0.5 * test_actions[:, 0] + 0.2 * test_states[:, 0] * test_states[:, 1]
    )
    
    # Plot results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot dimension 1: s₁ + a
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(test_states[:, 0], test_states[:, 1], true_next[:, 0], 
               c='green', label='True')
    ax1.scatter(test_states[:, 0], test_states[:, 1], preds_mean[:, 0], 
               c='blue', label='Predicted')
    # Plot training points in red
    train_states_flat = states.reshape(-1, state_dim)
    train_next_flat = next_states.reshape(-1, state_dim)
    ax1.scatter(train_states_flat[:, 0], train_states_flat[:, 1], train_next_flat[:, 0],
                c='red', marker='.', alpha=0.5, label='Train')
    ax1.set_xlabel('State dim 1')
    ax1.set_ylabel('State dim 2')
    ax1.set_zlabel('Next state dim 1')
    ax1.set_title('Next State Dimension 1 (s₁ + a)')
    ax1.legend()
    
    # Plot dimension 2: s₂ - a
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(test_states[:, 0], test_states[:, 1], true_next[:, 1], 
               c='green', marker='o', label='True')
    ax2.scatter(test_states[:, 0], test_states[:, 1], preds_mean[:, 1], 
               c='blue', marker='^', label='Predicted')
    # Plot training points in red
    ax2.scatter(train_states_flat[:, 0], train_states_flat[:, 1], train_next_flat[:, 1],
                c='red', marker='.', alpha=0.5, label='Train')
    ax2.set_xlabel('State dim 1')
    ax2.set_ylabel('State dim 2')
    ax2.set_zlabel('Next state dim 2')
    ax2.set_title('Next State Dimension 2 (s₂ - a)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # Plot uncertainty for dimension 1 for a fixed second dimension
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(test_states[100:120, 0], true_next[100:120, 0], c='green', marker='o', label='True')
    ax.plot(test_states[100:120, 0], preds_mean[100:120, 0], c='blue', marker='^', label='Predicted')
    ax.fill_between(test_states[100:120, 0], 
                    preds_mean[100:120, 0] - 10* ep_std[100:120, 0], 
                    preds_mean[100:120, 0] + 10* ep_std[100:120, 0], 
                    color='blue', alpha=0.2, label='Uncertainty')
    # Plot training points in red
    # Filter training points with similar second dimension as test_states[100:120, 1]
    target_s2 = test_states[100:120, 1].mean()
    mask = jnp.abs(states[:, 1] - target_s2) < 0.1
    

    filtered_train_states = states[mask]
    filtered_train_next = next_states[mask]
    ax.scatter(filtered_train_states[:, 0], filtered_train_next[:, 0], c='red', marker='.', alpha=0.5, label='Train')
    ax.set_xlabel('State dim 1')
    ax.set_ylabel('Next state dim 1')
    ax.set_title('Next State Dimension 1 with Uncertainty')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()