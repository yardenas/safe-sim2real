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


def update_fn(
    policy_loss_fn,
    value_loss_fn,
    cost_value_loss_fn,
    optimizer,
    value_optimizer,
    cost_value_optimizer,
    planning_env,
    unroll_length,
    num_minibatches,
    make_policy,
    num_updates_per_batch,
    batch_size,
    num_envs,
    env_step_per_training_step,
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
        optimizer_state, params, penalizer_params, key = carry
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

        (_, cost_value_aux), cost_value_params, cost_value_optimizer_state = cost_value_gradient_update_fn(
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
        return (optimizer_state, params, penalizer_params, key), aux

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, penalizer_params, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, penalizer_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, penalizer_params, key), aux

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

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

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            generate_unroll = lambda state: acting.generate_unroll(
                planning_env,
                state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            next_state, data = generate_unroll(current_state)
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)


        (optimizer_state, params, penalizer_params, _), aux = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=training_state.normalizer_params),
            (
                training_state.optimizer_state,
                training_state.params,
                training_state.penalizer_params,
                key_sgd,
            ),
            (),
            length=num_updates_per_batch,
        )
        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=training_state.normalizer_params,
            penalizer_params=penalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )  # type: ignore
        
        return (new_training_state, state, new_key), aux

    return training_step
