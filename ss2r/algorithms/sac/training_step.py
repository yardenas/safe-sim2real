from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.training.types import PRNGKey

from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    TrainingState,
    Transition,
)


def make_training_step(
    env,
    make_policy,
    replay_buffer,
    alpha_update,
    critic_update,
    cost_critic_update,
    actor_update,
    safe,
    min_alpha,
    reward_q_transform,
    cost_q_transform,
    penalizer,
    grad_updates_per_step,
    extra_fields,
    get_experience_fn,
    env_steps_per_experience_call,
    safety_budget,
    tau,
    num_critic_updates_per_actor_update,
    critic_entropy=True,
):
    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey, int], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey, int], Metrics]:
        training_state, key, count = carry

        key, key_alpha, key_critic, key_cost_critic, key_actor = jax.random.split(
            key, 5
        )
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        critic_loss, qr_params, qr_optimizer_state = critic_update(
            training_state.qr_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qr_params,
            alpha if critic_entropy else 0.0,
            transitions,
            key_critic,
            reward_q_transform,
            optimizer_state=training_state.qr_optimizer_state,
            params=training_state.qr_params,
        )
        if safe:
            cost_critic_loss, qc_params, qc_optimizer_state = cost_critic_update(
                training_state.qc_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_qc_params,
                alpha,
                transitions,
                key_critic,
                cost_q_transform,
                True,
                optimizer_state=training_state.qc_optimizer_state,
                params=training_state.qc_params,
            )
            cost_metrics = {
                "cost_critic_loss": cost_critic_loss,
            }
        else:
            cost_metrics = {}
            qc_params = None
            qc_optimizer_state = None

        # TODO (yarden): try to make it faster with cond later
        (actor_loss, aux), new_policy_params, new_policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.qr_params,
            training_state.qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            penalizer,
            training_state.penalizer_params,
            optimizer_state=training_state.policy_optimizer_state,
            params=training_state.policy_params,
        )
        should_update_actor = count % num_critic_updates_per_actor_update == 0
        update_if_needed = lambda x, y: jnp.where(should_update_actor, x, y)
        policy_params = jax.tree_map(
            update_if_needed, new_policy_params, training_state.policy_params
        )
        policy_optimizer_state = jax.tree_map(
            update_if_needed,
            new_policy_optimizer_state,
            training_state.policy_optimizer_state,
        )
        polyak = lambda target, new: jax.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)
        if safe:
            new_target_qc_params = polyak(training_state.target_qc_params, qc_params)
        else:
            new_target_qc_params = None
        if aux:
            new_penalizer_params = aux.pop("penalizer_params")
            additional_metrics = {
                **aux,
            }
        else:
            new_penalizer_params = training_state.penalizer_params
            additional_metrics = {}

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            **cost_metrics,
            **additional_metrics,
        }
        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            qr_optimizer_state=qr_optimizer_state,
            qc_optimizer_state=qc_optimizer_state,
            qr_params=qr_params,
            qc_params=qc_params,
            target_qr_params=new_target_qr_params,
            target_qc_params=new_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            penalizer_params=new_penalizer_params,
        )  # type: ignore
        return (new_training_state, key, count + 1), metrics

    def run_experience_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        """Runs the non-jittable experience collection step."""
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience_fn(
            env,
            make_policy,
            training_state.policy_params,
            training_state.normalizer_params,
            replay_buffer,
            env_state,
            buffer_state,
            experience_key,
            extra_fields,
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_experience_call,
        )
        return training_state, env_state, buffer_state, training_key

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        """Splits training into experience collection and a jitted training step."""
        training_state, env_state, buffer_state, training_key = run_experience_step(
            training_state, env_state, buffer_state, key
        )
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        metrics |= env_state.metrics
        return training_state, env_state, buffer_state, metrics

    return training_step
