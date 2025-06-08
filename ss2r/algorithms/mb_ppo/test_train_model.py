from typing import Mapping

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from brax.training import gradients
from brax.training.acme import running_statistics, specs
from brax.training.types import Transition

from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.mb_ppo import networks as mb_ppo_networks


def create_env_data(env, key, num_traj, traj_len):
    """Create a dummy environment data for testing."""

    def rollout_step(carry, _):
        obs, key = carry
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (env.action_size,), minval=-1.0, maxval=1.0)
        next_transition = env.step(obs, action)
        # Gather data for this step
        data = (
            obs.obs,
            action,
            next_transition.obs,
            next_transition.reward,
            0.5 * jnp.sum(jnp.abs(action)),
        )
        return (next_transition, key), data

    def single_traj(key):
        (final_obs, _), data = jax.lax.scan(
            rollout_step, (env.reset(key), key), None, traj_len
        )
        return data

    traj_keys = jax.random.split(key, num_traj)
    data = jax.vmap(single_traj)(traj_keys)
    data = [x.reshape(-1, *x.shape[2:]) for x in data]

    # Convert data to arrays
    obs, actions, next_obs, rewards, costs = data

    return obs, actions, next_obs, rewards, costs


def create_transitions(obs, actions, next_obs, rewards, costs):
    # Create extras dictionary with state_extras and policy_extras
    extras = {
        "state_extras": {
            "truncation": jnp.zeros(obs.shape[0]),  # No truncation in offline data
            "cost": costs,  # Include costs in state_extras
        },
        "policy_extras": {
            "log_prob": jnp.zeros(obs.shape[0]),  # Not needed for model training
            "raw_action": actions,  # Same as actions for model training
        },
    }
    discounts = jnp.ones_like(rewards)
    transitions = Transition(
        observation=obs,
        action=actions,
        reward=rewards,
        discount=discounts,
        next_observation=next_obs,
        extras=extras,
    )

    return transitions


def create_loss_and_networks(
    obs_size,
    action_size,
    normalize_fn,
    denormalize_fn,
    num_ensemble,
    lr,
    init_key,
):
    """Create the loss function and networks for model-based PPO."""
    ppo_network = mb_ppo_networks.make_mb_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        postprocess_observations_fn=denormalize_fn,
        n_ensemble=num_ensemble,
        model_hidden_layer_sizes=[512, 512],
        activation=jax.nn.swish,
        use_bro=False,
        learn_std=False,
    )

    init_model_ensemble = jax.vmap(ppo_network.model_network.init)
    model_keys = jax.random.split(init_key, 5)
    params = init_model_ensemble(model_keys)

    # Make loss
    model_loss, *_ = mb_ppo_losses.make_losses(
        ppo_network=ppo_network,
        entropy_cost=1e-4,
        discounting=0.9,
        safety_discounting=0.9,
        reward_scaling=1.0,
        cost_scaling=1.0,
        gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
        normalize_fn=normalize_fn,
    )

    model_optimizer = optax.adam(learning_rate=lr)
    optimizer_state = model_optimizer.init(params)

    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))

    normalizer_params = running_statistics.init_state(obs_shape)

    return (
        ppo_network,
        model_loss,
        params,
        model_optimizer,
        optimizer_state,
        normalizer_params,
    )


def train_model(
    transitions,
    model_loss,
    model_params,
    model_optimizer,
    optimizer_state,
    normalizer_params,
    epochs,
    key_sgd,
):
    """Train the model using synthetic transitions."""
    # Update normalizer params
    normalizer_params = running_statistics.update(
        normalizer_params, transitions.observation
    )

    # Make gradient function
    model_update_fn = gradients.gradient_update_fn(
        model_loss,
        model_optimizer,
        pmap_axis_name=None,
        has_aux=True,
    )

    def sgd_step(carry, _):
        params, optimizer_state, transitions, key_sgd = carry
        key_sgd, sample_key = jax.random.split(key_sgd, 2)
        # Sample a batch of transitions of size 100
        transitions_batch = jax.tree.map(
            lambda x: jax.random.permutation(sample_key, x)[:256], transitions
        )
        (_, aux), new_params, new_optimizer_state = model_update_fn(
            params,
            normalizer_params,
            transitions_batch,
            False,
            optimizer_state=optimizer_state,  # type: ignore
        )
        return (new_params, new_optimizer_state, transitions, key_sgd), aux

    def epoch_step(carry, _):
        params, optimizer_state, transitions, normalizer_params, key_sgd = carry
        key_sgd, subkey = jax.random.split(key_sgd)

        (params, optimizer_state, transitions, key_sgd), metrics = jax.lax.scan(
            sgd_step, (params, optimizer_state, transitions, subkey), None, length=1
        )

        return (
            params,
            optimizer_state,
            transitions,
            normalizer_params,
            key_sgd,
        ), metrics

    def train_model(
        params, optimizer_state, transitions, normalizer_params, key_sgd, num_epochs
    ):
        carry = (params, optimizer_state, transitions, normalizer_params, key_sgd)
        carry, metrics = jax.lax.scan(epoch_step, carry, None, length=num_epochs)
        return carry, metrics

    (
        (new_params, new_optimizer_state, transitions, normalizer_params, key_sgd),
        metrics,
    ) = train_model(
        model_params,
        optimizer_state,
        transitions,
        normalizer_params,
        key_sgd,
        num_epochs=epochs,
    )

    return new_params, new_optimizer_state, metrics


def compare_trajectories(
    key,
    env,
    ppo_network,
    planning_env,
    normalizer_params,
    model_params,
    num_timesteps=15,
    num_trajs=5,
):
    key, rng_key = jax.random.split(key)
    keys = jax.random.split(key, num_trajs)
    initial_states_env = jax.vmap(env.reset)(keys)

    def rollout_env(carry, _):
        state_env, key = carry
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (env.action_size,), minval=-1.0, maxval=1.0)
        next_state_env = env.step(state_env, action)
        return (next_state_env, key), (
            next_state_env.obs,
            next_state_env.reward,
            action,
        )

    def rollout_model_env(carry, action):
        state_model = carry
        return planning_env.step(state_model, action)

    def single_traj_model_env(init_state_env, actions):
        (final_state, _), (states) = jax.lax.scan(
            rollout_model_env, init_state_env, actions[1:], length=num_timesteps
        )
        obs_seq = jnp.concatenate([init_state_env.obs[None, ...], states.obs], axis=0)
        reward_seq = jnp.concatenate([jnp.ones((1,)), states.reward], axis=0)
        cost_seq = jnp.concatenate([jnp.zeros((1,)), states.cost], axis=0)
        return obs_seq, reward_seq, cost_seq

    def single_env_traj(init_state_env, key):
        (final_state, _), (obs_seq, reward_seq, action_seq) = jax.lax.scan(
            rollout_env, (init_state_env, key), None, length=num_timesteps
        )
        obs_seq = jnp.concatenate([init_state_env.obs[None, ...], obs_seq], axis=0)
        reward_seq = jnp.concatenate([jnp.ones((1,)), reward_seq], axis=0)
        action_seq = jnp.concatenate(
            [jnp.zeros((1, env.action_size)), action_seq], axis=0
        )
        return obs_seq, reward_seq, action_seq

    obs_env, rewards_env, actions_env = jax.vmap(single_env_traj)(
        initial_states_env, keys
    )

    initial_states_env.info = (
        {
            "cumulative_cost": jnp.zeros_like(initial_states_env.reward),  # type: ignore
            "truncation": jnp.zeros_like(initial_states_env.reward),
            "cost": jnp.zeros_like(initial_states_env.reward),
            "key": jnp.tile(rng_key[None], (initial_states_env.obs.shape[0], 1)),
        },
    )

    obs_model_env, rewards_model_env = jax.vmap(single_traj_model_env)(
        initial_states_env, actions_env
    )

    obs_env = jnp.array(obs_env)
    obs_model = jnp.array(obs_model_env)
    rewards_env = jnp.array(rewards_env)
    rewards_model = jnp.array(rewards_model_env)

    plt.figure(figsize=(18, 10))
    for traj in range(num_trajs):
        for i in range(min(5, obs_env.shape[2])):  # up to 5 state dims
            plt.subplot(num_trajs, 6, traj * 6 + i + 1)
            plt.plot(obs_env[traj, :, i], label="Env", color="blue")
            plt.plot(obs_model[traj, :, i], label="Model", color="red", linestyle="--")
            plt.plot(
                obs_model_env[traj, :, i],
                label="Model Env",
                color="green",
                linestyle=":",
            )
            plt.title(f"Traj {traj+1} - State {i}")
            if traj == 0:
                plt.legend()
        plt.subplot(num_trajs, 6, traj * 6 + 6)
        plt.plot(rewards_env[traj], label="Env", color="blue")
        plt.plot(rewards_model[traj], label="Model", color="red", linestyle="--")
        plt.plot(
            rewards_model_env[traj], label="Model Env", color="green", linestyle=":"
        )
        plt.title(f"Traj {traj+1} - Reward")
        if traj == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig("trajectory_comparison_.png")
