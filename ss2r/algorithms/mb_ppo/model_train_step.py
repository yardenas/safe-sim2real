import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey

from ss2r.algorithms.mb_ppo import _PMAP_AXIS_NAME, Metrics, TrainingState
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.sac.types import ReplayBufferState, float32

def update_fn(
    model_loss_fn,
    model_optimizer, 
    replay_buffer,
    num_minibatches,
    num_updates_per_batch,
    learn_std,
):

    model_update_fn = gradients.gradient_update_fn(
        model_loss_fn,
        model_optimizer,
        pmap_axis_name=_PMAP_AXIS_NAME,
        has_aux=True,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        (
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        ) = optimizer_state

        key, key_loss = jax.random.split(key)
        (_,aux), model_params, model_optimizer_state = model_update_fn(
            params.model,
            normalizer_params,
            data,
            key_loss,
            learn_std,
            optimizer_state=model_optimizer_state,
        )

        optimizer_state = (
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        )

        params = mb_ppo_losses.MBPPOParams(
            model_params, params.policy, params.value, params.cost_value
        )  # type: ignore
        return (optimizer_state, params, key), aux

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x):
            # Skip reshaping for non-array elements or null elements
            if not hasattr(x, "shape"):
                return x
                
            # Permute the data first
            x = jax.random.permutation(key_perm, x)
            
            # Get the total number of samples
            total_samples = x.shape[0]
            
            # If we have fewer samples than minibatches, we need to adjust our approach
            if total_samples < num_minibatches:
                # Option 1: Repeat the data to fill minibatches
                repeat_factor = (num_minibatches + total_samples - 1) // total_samples  # Ceiling division
                x = jnp.repeat(x, repeat_factor, axis=0)[:num_minibatches]
                # Reshape to (num_minibatches, 1) + rest of dimensions
                return jnp.reshape(x, (num_minibatches, 1) + x.shape[1:])
            
            # Calculate how many samples should go into each minibatch
            samples_per_minibatch = total_samples // num_minibatches
            
            # Now reshape with the calculated dimensions
            x = jnp.reshape(x, (num_minibatches, samples_per_minibatch) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), aux

    def training_step(
        carry: Tuple[TrainingState, ReplayBufferState, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, buffer_state, training_key = carry
        key_sgd, new_key = jax.random.split(training_key, 2)

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = float32(transitions)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_updates_per_batch, -1) + x.shape[1:]),
            transitions,
        )

        (optimizer_state, params, _), aux = jax.lax.scan(
            functools.partial(sgd_step, data=transitions, normalizer_params=training_state.normalizer_params),
            (
                training_state.optimizer_state,
                training_state.params,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )
        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=training_state.normalizer_params,
            env_steps=training_state.env_steps,
        )  # type: ignore
        
        return (new_training_state, buffer_state, new_key), aux

    return training_step
