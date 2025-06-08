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
    make_policy,
    make_model_env,
    model_replay_buffer,
    sac_replay_buffer,
    alpha_update,
    critic_update,
    model_update,
    actor_update,
    min_alpha,
    reward_q_transform,
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
    model_to_real_data_ratio,
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
        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)
        metrics = {
            "critic_loss": critic_loss,
        }
        new_training_state = training_state.replace(  # type: ignore
            qr_optimizer_state=qr_optimizer_state,
            qr_params=qr_params,
            target_qr_params=new_target_qr_params,
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
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (model_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
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
            make_policy,
            training_state.policy_params,
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
        transitions: Transition,
        sac_replay_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> ReplayBufferState:
        key_generate_unroll, cost_key, model_key, key_perm = jax.random.split(key, 4)
        assert (
            num_model_rollouts
            <= transitions.observation.shape[0] * transitions.observation.shape[1]
        ), "num_model_rollouts must be less than or equal to the number of transitions"
        transitions = float32(transitions)
        cumulative_cost = jax.random.uniform(
            cost_key, (transitions.reward.shape[0],), minval=0.0, maxval=0.0
        )
        state = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=jnp.zeros_like(transitions.reward),
            info={
                "cumulative_cost": cumulative_cost,  # type: ignore
                "truncation": jnp.zeros_like(cumulative_cost),
                "cost": transitions.extras["state_extras"].get(
                    "cost", jnp.zeros_like(cumulative_cost)
                ),
                "key": jnp.tile(model_key[None], (transitions.observation.shape[0], 1)),
            },
        )
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
    ) -> Transition:
        pred_fn = planning_env.model_network.apply
        model_params = planning_env.model_params
        normalizer_params = planning_env.normalizer_params
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward, cost = vmap_pred_fn(
            normalizer_params, model_params, transitions.observation, transitions.action
        )
        disagreement = next_obs_pred.std(axis=0).mean(-1)
        new_reward = reward.mean(0) + disagreement * optimism
        next_obs_pred = next_obs_pred.mean(0)
        return Transition(
            observation=transitions.observation,
            next_observation=next_obs_pred,
            action=transitions.action,
            reward=new_reward,
            discount=transitions.discount,
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
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        (training_state, _), model_metrics = jax.lax.scan(
            model_sgd_step, (training_state, training_key), transitions
        )
        planning_env = make_model_env(
            model_params=training_state.model_params,
            normalizer_params=training_state.normalizer_params,
        )
        planning_env = VmapWrapper(planning_env)
        policy = make_policy(
            (training_state.normalizer_params, training_state.policy_params)
        )
        # Rollout trajectories from the sampled transitions
        sac_buffer_state = generate_model_data(
            planning_env, policy, transitions, sac_buffer_state, training_key
        )
        # Train SAC with model data
        sac_buffer_state, model_transitions = sac_replay_buffer.sample(sac_buffer_state)
        num_real_transitions = int(
            model_transitions.reward.shape[0] * (1 - model_to_real_data_ratio)
        )
        assert (
            num_real_transitions <= transitions.reward.shape[0]
        ), "More model minibatches than real minibatches"
        transitions = jax.tree_util.tree_map(
            lambda x, y: x.at[:num_real_transitions].set(y[:num_real_transitions]),
            model_transitions,
            transitions,
        )
        model_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (critic_grad_updates_per_step, -1) + x.shape[1:]),
            model_transitions,
        )
        transitions, disagreement = relabel_transitions(planning_env, transitions)
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
