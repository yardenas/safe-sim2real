from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import VmapWrapper
from brax.training import acting
from brax.training.types import Policy, PRNGKey

from ss2r.algorithms.mbpo.model_env import ModelBasedEnv
from ss2r.algorithms.mbpo.types import TrainingState
from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
    float32,
)


def make_training_step(
    env,
    make_planning_policy,
    make_rollout_policy,
    get_rollout_policy_params,
    make_model_env,
    model_replay_buffer,
    sac_replay_buffer,
    alpha_update,
    critic_update,
    cost_critic_update,
    model_update,
    actor_update,
    safe,
    min_alpha,
    reward_q_transform,
    cost_q_transform,
    model_grad_updates_per_step,
    critic_grad_updates_per_step,
    extra_fields,
    get_experience_fn,
    env_steps_per_experience_call,
    tau,
    num_critic_updates_per_actor_update,
    unroll_length,
    num_model_rollouts,
    optimism,
    pessimism,
    model_to_real_data_ratio,
    scaling_fn,
    reward_termination,
    use_termination,
):
    def critic_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        transitions = float32(transitions)
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        critic_loss, qr_params, qr_optimizer_state = critic_update(
            training_state.qr_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qr_params,
            alpha,
            transitions,
            key_critic,
            reward_q_transform,
            optimizer_state=training_state.qr_optimizer_state,
            params=training_state.qr_params,
        )
        if safe:
            cost_critic_loss, qc_params, qc_optimizer_state = cost_critic_update(
                training_state.qc_params,
                training_state.backup_policy_params,
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
            qc_params = training_state.qc_params
            qc_optimizer_state = training_state.qc_optimizer_state
            cost_metrics = {
                "cost_critic_loss": cost_critic_loss,
            }
        else:
            cost_metrics = {}
            qc_params = None
            qc_optimizer_state = None

        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)
        if safe:
            new_target_qc_params = polyak(training_state.target_qc_params, qc_params)
        else:
            new_target_qc_params = None
        metrics = {
            "critic_loss": critic_loss,
            "fraction_done": 1.0 - transitions.discount.mean(),
            **cost_metrics,
        }
        new_training_state = training_state.replace(  # type: ignore
            qr_optimizer_state=qr_optimizer_state,
            qr_params=qr_params,
            qc_optimizer_state=qc_optimizer_state,
            qc_params=qc_params,
            target_qr_params=new_target_qr_params,
            target_qc_params=new_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
        )
        return (new_training_state, key), metrics

    def actor_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_actor = jax.random.split(key, 3)
        transitions = float32(transitions)
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.qr_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
            params=training_state.policy_params,
        )
        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
        }
        new_training_state = training_state.replace(  # type: ignore
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
        )
        return (new_training_state, key), metrics

    def model_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        # TODO (yarden): can remove this
        key, _ = jax.random.split(key)
        transitions = float32(transitions)
        model_loss, model_params, model_optimizer_state = model_update(
            training_state.model_params,
            training_state.normalizer_params,
            transitions,
            optimizer_state=training_state.model_optimizer_state,  # type: ignore
            params=training_state.model_params,
        )
        new_training_state = training_state.replace(  # type: ignore
            model_optimizer_state=model_optimizer_state,
            model_params=model_params,
        )
        metrics = {"model_loss": model_loss}
        return (new_training_state, key), metrics

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
            make_rollout_policy,
            get_rollout_policy_params(training_state),
            training_state.normalizer_params,
            model_replay_buffer,
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

    def generate_model_data(
        planning_env: ModelBasedEnv,
        policy: Policy,
        sac_replay_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> ReplayBufferState:
        key_generate_unroll, cost_key, model_key, key_perm = jax.random.split(key, 4)
        keys = jax.random.split(key, num_model_rollouts + 2)
        key = keys[0]
        key_generate_unroll = keys[1]
        rollout_keys = keys[2:]
        state = planning_env.reset(rollout_keys)
        _, transitions = acting.generate_unroll(
            planning_env,
            state,
            policy,
            key_generate_unroll,
            unroll_length,
            extra_fields=extra_fields,
        )
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
        sac_replay_buffer_state = sac_replay_buffer.insert(
            sac_replay_buffer_state, float16(transitions)
        )
        return sac_replay_buffer_state

    def relabel_transitions(
        planning_env: ModelBasedEnv,
        transitions: Transition,
        key: PRNGKey,
    ) -> Transition:
        pred_fn = planning_env.model_network.apply
        model_params = planning_env.model_params
        normalizer_params = planning_env.normalizer_params
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward, cost = vmap_pred_fn(
            normalizer_params, model_params, transitions.observation, transitions.action
        )
        disagreement = (
            next_obs_pred.std(axis=0).mean(-1)
            if isinstance(next_obs_pred, jax.Array)
            else next_obs_pred["state"].std(axis=0).mean(-1)
        )
        new_reward = reward.mean(0) + disagreement * optimism
        discount = transitions.discount
        if safe:
            cost = cost.mean(0) + disagreement * pessimism
            transitions.extras["state_extras"]["cost"] = cost
            if (planning_env.qc_network is not None) and use_termination:
                qc_pred = planning_env.qc_network.apply(
                    normalizer_params,
                    planning_env.qc_params,
                    transitions.observation,
                    transitions.action,
                )
                curr_discount = (
                    transitions.observation["curr_discount"]
                    * planning_env.cost_discount
                )
                expected_total_cost = qc_pred.mean(
                    axis=-1
                ) * curr_discount.squeeze() + scaling_fn(
                    transitions.observation["cumulative_cost"].squeeze()
                )
                discount = jnp.where(
                    (expected_total_cost > planning_env.safety_budget)
                    & jax.random.bernoulli(
                        key,
                        p=planning_env.termination_prob,
                        shape=expected_total_cost.shape,
                    ),
                    jnp.zeros_like(cost, dtype=jnp.float32),
                    jnp.ones_like(cost, dtype=jnp.float32),
                )
        new_reward = jnp.where(
            discount,
            new_reward,
            jnp.ones_like(new_reward) * reward_termination,
        )
        next_obs_pred = jax.tree_map(lambda x: x.mean(0), next_obs_pred)
        return Transition(
            observation=transitions.observation,
            next_observation=next_obs_pred,
            action=transitions.action,
            reward=new_reward,
            discount=discount,
            extras=transitions.extras,
        ), disagreement

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        model_buffer_state: ReplayBufferState,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        """Splits training into experience collection and a jitted training step."""
        (
            training_state,
            env_state,
            model_buffer_state,
            training_key,
        ) = run_experience_step(training_state, env_state, model_buffer_state, key)
        model_buffer_state, transitions = model_replay_buffer.sample(model_buffer_state)
        assert (
            num_model_rollouts <= transitions.reward.shape[0]
        ), "num_model_rollouts must be less than or equal to the number of transitions"
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        tmp_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (model_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), model_metrics = jax.lax.scan(
            model_sgd_step, (training_state, training_key), tmp_transitions
        )
        planning_env = make_model_env(
            model_params=training_state.model_params,
            qc_params=training_state.qc_params,
            normalizer_params=training_state.normalizer_params,
            transitions=transitions,
        )
        planning_env = VmapWrapper(planning_env)
        policy = make_planning_policy(
            (training_state.normalizer_params, training_state.policy_params)
        )
        # Rollout trajectories from the sampled transitions
        sac_buffer_state = generate_model_data(
            planning_env, policy, sac_buffer_state, training_key
        )
        # Train SAC with model data
        sac_buffer_state, model_transitions = sac_replay_buffer.sample(sac_buffer_state)
        num_real_transitions = int(
            model_transitions.reward.shape[0] * (1 - model_to_real_data_ratio)
        )
        assert (
            num_real_transitions <= transitions.reward.shape[0]
        ), "More model minibatches than real minibatches"
        if num_real_transitions >= 1:
            transitions = jax.tree_util.tree_map(
                lambda x, y: x.at[:num_real_transitions].set(y[:num_real_transitions]),
                model_transitions,
                transitions,
            )
        else:
            transitions = model_transitions
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (critic_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        transitions, disagreement = relabel_transitions(
            planning_env, transitions, training_key
        )
        (training_state, _), critic_metrics = jax.lax.scan(
            critic_sgd_step, (training_state, training_key), transitions
        )
        num_actor_updates = -(
            -critic_grad_updates_per_step // num_critic_updates_per_actor_update
        )
        assert num_actor_updates > 0, "Actor updates is non-positive"
        transitions = jax.tree_util.tree_map(
            lambda x: x[:num_actor_updates], transitions
        )
        (training_state, _), actor_metrics = jax.lax.scan(
            actor_sgd_step,
            (training_state, training_key),
            transitions,
            length=num_actor_updates,
        )
        metrics = {**model_metrics, **critic_metrics, **actor_metrics}
        metrics["buffer_current_size"] = model_replay_buffer.size(model_buffer_state)
        metrics |= env_state.metrics
        metrics["disagreement"] = disagreement
        return training_state, env_state, model_buffer_state, metrics

    return training_step
