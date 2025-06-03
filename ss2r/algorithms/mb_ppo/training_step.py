import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import acting, gradients, types
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey

from ss2r.algorithms.mb_ppo import Metrics, TrainingState
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.sac.types import float32  # ,ReplayBufferState


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
    safe,
    env,
):
    policy_gradient_update_fn = gradients.gradient_update_fn(
        policy_loss_fn, optimizer, pmap_axis_name=None, has_aux=True
    )
    value_gradient_update_fn = gradients.gradient_update_fn(
        value_loss_fn, value_optimizer, pmap_axis_name=None, has_aux=True
    )
    cost_value_gradient_update_fn = gradients.gradient_update_fn(
        cost_value_loss_fn,
        cost_value_optimizer,
        pmap_axis_name=None,
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
        if safe:
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
        else:
            cost_value_params = params.cost_value
        optimizer_state = (
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        )
        params = mb_ppo_losses.MBPPOParams(
            policy_params, value_params, cost_value_params, params.model
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
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, buffer_state, key = carry
        key_sgd, key_generate_unroll, cost_key, new_key = jax.random.split(key, 4)

        # Create planning environment with current model parameters
        # planning_env = planning_env_factory(
        #     training_state.params.model, training_state.normalizer_params
        # )
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

        # Function to generate unrolls from each initial state
        def f(carry, unused_t):
            # Sample initial states from the replay buffer
            current_buffer_state, current_key = carry

            current_buffer_state, transitions = replay_buffer.sample(
                current_buffer_state
            )
            transitions = float32(transitions)

            # FIXME (manu): make sure that minval and maxval are correct
            # cumulative_cost = jax.random.uniform(
            #     cost_key, (transitions.reward.shape[0],), minval=0.0, maxval=0.0
            # )
            # FIXME (manu): make sure to delete this again
            reset_fn = env.reset
            batch_size = replay_buffer._sample_batch_size
            keys = jax.random.split(current_key, batch_size + 1)
            sample_keys = keys[:batch_size].reshape(batch_size, -1)
            current_key = keys[-1]
            env_state = reset_fn(sample_keys)

            env_state = env_state.replace(
                pipeline_state=transitions.extras["state_extras"][
                    "pipeline_state"
                ],  # REDO!!
                obs=transitions.observation,
                reward=transitions.reward,
                done=transitions.discount,
            )

            # state = envs.State(
            #     pipeline_state=v_init, #REDO!!
            #     obs=transitions.observation,
            #     reward=transitions.reward,
            #     done=transitions.discount,
            #     info={
            #         "cumulative_cost": cumulative_cost,  # type: ignore
            #         "truncation": transitions.extras["state_extras"]["truncation"],
            #         "cost": transitions.extras["state_extras"].get(
            #             "cost", jnp.zeros_like(cumulative_cost)
            #         ),
            #     },
            # )
            next_key, current_key = jax.random.split(current_key)
            generate_unroll = lambda state: acting.generate_unroll(
                env,  # REDO!!
                state,
                policy,
                current_key,
                unroll_length,
                extra_fields=extra_fields,
            )
            _, data = generate_unroll(env_state)
            return (current_buffer_state, next_key), data

        (buffer_state, _), data = jax.lax.scan(
            f,
            (buffer_state, key_generate_unroll),
            (),
            length=num_minibatches,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
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
