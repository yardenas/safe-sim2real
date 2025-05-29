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
from ss2r.algorithms.sac.types import ReplayBufferState


def update_fn(
    policy_loss_fn,
    value_loss_fn,
    cost_value_loss_fn,
    optimizer,
    value_optimizer,
    cost_value_optimizer,
    planning_env_factory,  # Changed from planning_env to planning_env_factory
    replay_buffer,
    unroll_length,
    num_minibatches,
    make_policy,
    num_updates_per_batch,
    batch_size,
    safe,
):
    policy_gradient_update_fn = gradients.gradient_update_fn(
        policy_loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    value_gradient_update_fn = gradients.gradient_update_fn(
        value_loss_fn, value_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    cost_value_gradient_update_fn = gradients.gradient_update_fn(
        cost_value_loss_fn,
        cost_value_optimizer,
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
        (_, aux), policy_params, policy_optimizer_state = policy_gradient_update_fn(
            params.policy,
            params.value,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=policy_optimizer_state,
        )

        (_, value_aux), value_params, value_optimizer_state = value_gradient_update_fn(
            params.value,
            normalizer_params,
            data,
            optimizer_state=value_optimizer_state,
        )
        aux |= value_aux

        (
            (_, cost_value_aux),
            cost_value_params,
            cost_value_optimizer_state,
        ) = cost_value_gradient_update_fn(
            params.cost_value,
            normalizer_params,
            data,
            optimizer_state=cost_value_optimizer_state,
        )
        aux |= cost_value_aux

        optimizer_state = (
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        )
        params = mb_ppo_losses.MBPPOParams(
            params.model, policy_params, value_params, cost_value_params
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

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
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
    ) -> Tuple[Tuple[TrainingState, ReplayBufferState, PRNGKey], Metrics]:
        training_state, buffer_state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        # Create planning environment with current model parameters
        planning_env = planning_env_factory(
            training_state.params.model, training_state.normalizer_params
        )

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        extra_fields = ("truncation",)
        if safe:
            extra_fields += ("cost", "cumulative_cost")  # type: ignore

        # Sample initial states from the replay buffer
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        initial_states = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=transitions.discount,
            info={},
        )

        # Function to generate unrolls from each initial state
        def generate_unroll_fn(state):
            return acting.generate_unroll(
                planning_env,
                state,
                policy,
                key_generate_unroll,
                unroll_length,
                extra_fields=extra_fields,
            )

        # Generate all unrolls in parallel
        _, data = jax.vmap(generate_unroll_fn)(initial_states)

        # Print shapes of key data structures before reshape
        print(
            f"Data dimensions: {data.observation.shape}, {data.action.shape}, {data.reward.shape}, {data.discount.shape}, {data.extras['state_extras']['truncation'].shape}"
        )
        # Reshape reward, discount and truncation to get rid of trailing dimension
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        print(
            f"Data dimensions: {data.observation.shape}, {data.action.shape}, {data.reward.shape}, {data.discount.shape}, {data.extras['state_extras']['truncation'].shape}"
        )

        assert data.discount.shape[1:] == (unroll_length,)

        (optimizer_state, params, _), aux = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=training_state.normalizer_params
            ),
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
