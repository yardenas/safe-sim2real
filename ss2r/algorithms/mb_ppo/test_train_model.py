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
from ss2r.benchmark_suites.brax.cartpole import cartpole


def create_env_data(key, num_samples=1000):
    """Create a dummy environment data for testing."""
    env = cartpole.Cartpole(backend="generalized", swingup=False, sparse=False)

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
            rollout_step, (env.reset(key), key), None, length=200
        )
        return data

    traj_keys = jax.random.split(key, num_samples)
    data = jax.vmap(single_traj)(traj_keys)
    data = [x.reshape(-1, *x.shape[2:]) for x in data]

    # Convert data to arrays
    obs, actions, next_obs, rewards, costs = data

    return env, obs, actions, next_obs, rewards, costs


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


def create_synthetic_training_data(key, obs_size, actions_size, num_samples=1000):
    """Generate synthetic data for testing."""
    obs_key, actions_key = jax.random.split(key, 2)
    obs = jax.random.uniform(obs_key, (num_samples, obs_size), minval=-1, maxval=1)
    actions = jax.random.uniform(
        actions_key, (num_samples, actions_size), minval=-1, maxval=1
    )
    next_obs = obs
    next_obs = next_obs.at[:, 0].set(
        obs[:, 0] + jnp.sin(actions[:, 0]) * 0.2 + 0.05 * obs[:, 1] ** 2
    )
    next_obs = next_obs.at[:, 1].set(
        obs[:, 1] + jnp.cos(actions[:, 1]) * 0.15 - 0.03 * obs[:, 0] * actions[:, 1]
    )
    reward = (
        2.0
        - jnp.sum(jnp.square(obs), axis=1)
        - 0.1 * jnp.sum(jnp.square(actions), axis=1)
        + 0.05 * obs[:, 0] * actions[:, 1]
    )
    cost = (
        0.5 * jnp.sum(jnp.abs(actions), axis=1)
        + 0.2 * jnp.abs(obs[:, 0] * actions[:, 0])
        + 0.1 * jnp.abs(obs[:, 1] - actions[:, 1])
    )
    return obs, actions, next_obs, reward, cost


def test_train_model():
    num_samples = 1000

    key = jax.random.PRNGKey(0)
    gen_key, init_key, key_sgd = jax.random.split(key, 3)

    # Generate synthetic training data
    # obs, actions, next_obs, rewards, costs = create_synthetic_training_data(gen_key, obs_size, action_size, num_samples)
    env, obs, actions, next_obs, rewards, costs = create_env_data(gen_key, num_samples)
    obs_size = obs.shape[1] if isinstance(obs, jnp.ndarray) else obs.shape
    action_size = (
        actions.shape[1] if isinstance(actions, jnp.ndarray) else actions.shape
    )

    normalize_fn = running_statistics.normalize
    denormalize_fn = running_statistics.denormalize

    # Create a dummy model
    ppo_network = mb_ppo_networks.make_mb_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        postprocess_observations_fn=denormalize_fn,
        n_ensemble=1,
        model_hidden_layer_sizes=[512, 512],
        activation=jax.nn.swish,
        use_bro=False,
        learn_std=False,
    )

    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))

    normalizer_params = running_statistics.init_state(obs_shape)
    params = ppo_network.model_network.init(init_key)

    # Convert synthetic data
    transitions = create_transitions(obs, actions, next_obs, rewards, costs)
    model_optimizer = optax.adam(learning_rate=1e-3)
    optimizer_state = model_optimizer.init(params)

    # Update normalizer params
    normalizer_params = running_statistics.update(
        normalizer_params, transitions.observation
    )

    # Make loss
    model_loss, *_ = mb_ppo_losses.make_losses(
        ppo_network=ppo_network,
        preprocess_observations_fn=normalize_fn,
        entropy_cost=1e-4,
        discounting=0.9,
        safety_discounting=0.9,
        reward_scaling=1.0,
        cost_scaling=1.0,
        gae_lambda=0.95,
        safety_gae_lambda=0.95,
        clipping_epsilon=0.3,
        normalize_advantage=True,
    )

    # Make gradient function
    model_update_fn = gradients.gradient_update_fn(
        model_loss,
        model_optimizer,
        pmap_axis_name=None,
        has_aux=True,
    )

    def eval_model_mse(params, normalizer_params, transitions):
        """Evaluate the model using mean squared error."""
        model_apply = ppo_network.model_network.apply
        (
            (next_obs_pred, reward_pred, cost_pred),
            (next_obs_std, reward_std, cost_std),
        ) = model_apply(
            normalizer_params, params, transitions.observation, transitions.action
        )

        mse_next_obs = jnp.mean(
            jnp.square(next_obs_pred - transitions.next_observation)
        )
        mse_reward = jnp.mean(jnp.square(reward_pred - transitions.reward))
        mse_cost = jnp.mean(
            jnp.square(cost_pred - transitions.extras["state_extras"]["cost"])
        )

        return mse_next_obs, mse_reward, mse_cost

    def sgd_step(carry, _):
        params, optimizer_state, transitions, key_sgd = carry
        key_sgd, subkey, sample_key = jax.random.split(key_sgd, 3)
        # Sample a batch of transitions of size 100
        transitions_batch = jax.tree.map(
            lambda x: jax.random.permutation(sample_key, x)[:256], transitions
        )
        (_, aux), new_params, new_optimizer_state = model_update_fn(
            params,
            normalizer_params,
            transitions_batch,
            subkey,
            False,
            optimizer_state=optimizer_state,  # type: ignore
        )
        return (new_params, new_optimizer_state, transitions, key_sgd), aux

    def epoch_step(carry, _):
        params, optimizer_state, transitions, normalizer_params, key_sgd = carry
        key_sgd, subkey = jax.random.split(key_sgd)

        (params, optimizer_state, transitions, key_sgd), aux_hist = jax.lax.scan(
            sgd_step, (params, optimizer_state, transitions, key_sgd), None, length=100
        )

        mse_obs, mse_reward, mse_cost = eval_model_mse(
            params, normalizer_params, transitions
        )

        return (params, optimizer_state, transitions, normalizer_params, key_sgd), (
            mse_obs,
            mse_reward,
            mse_cost,
        )

    def train_model(
        params, optimizer_state, transitions, normalizer_params, key_sgd, num_epochs=5
    ):
        carry = (params, optimizer_state, transitions, normalizer_params, key_sgd)
        carry, mse_hist = jax.lax.scan(epoch_step, carry, None, length=num_epochs)
        return carry, mse_hist

    carry, mse_hist = train_model(
        params, optimizer_state, transitions, normalizer_params, key_sgd, num_epochs=50
    )

    plt.plot(mse_hist[0], label="MSE Next Obs")
    plt.plot(mse_hist[1], label="MSE Reward")
    plt.plot(mse_hist[2], label="MSE Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Model Training MSE")
    plt.legend()
    plt.savefig("model_training_mse.png")

    # print the average reward in carry
    transitions = carry[2]
    avg_reward = jnp.mean(transitions.reward)
    print(f"Average Reward: {avg_reward:.4f}")

    def compare_trajectories(env, normalizer_params, params, num_timesteps=15):
        states_env = []
        rewards_env = []
        states_model = []
        rewards_model = []
        current_state_env = env.reset(jax.random.PRNGKey(0))
        current_state_model = current_state_env.obs
        for _ in range(num_timesteps):
            action = jax.random.uniform(
                jax.random.PRNGKey(0), (env.action_size,), minval=-1.0, maxval=1.0
            )
            next_state_env = env.step(current_state_env, action)
            states_env.append(next_state_env.obs)
            rewards_env.append(next_state_env.reward)
            current_state_env = next_state_env

            (
                (next_state_model, reward_model, cost_model),
                _,
            ) = ppo_network.model_network.apply(
                normalizer_params, params, current_state_model, action
            )
            # Calculate mean of ensemble predictions
            next_state_model = jnp.mean(next_state_model, axis=0)
            reward_model = jnp.mean(reward_model, axis=0)
            cost_model = jnp.mean(cost_model, axis=0)
            states_model.append(next_state_model)
            rewards_model.append(reward_model)
            current_state_model = next_state_model

        states_env = jnp.array(states_env)
        states_model = jnp.array(states_model)
        rewards_env = jnp.array(rewards_env)
        rewards_model = jnp.array(rewards_model)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1)
        plt.plot(states_env[:, 0], label="Env States", color="blue")
        plt.plot(states_model[:, 0], label="Model States", color="red", linestyle="--")
        plt.title("State Comparison - State 0")
        plt.xlabel("Time Step")
        plt.ylabel("State 0")
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.plot(states_env[:, 1], label="Env States", color="blue")
        plt.plot(states_model[:, 1], label="Model States", color="red", linestyle="--")
        plt.title("State Comparison - State 1")
        plt.xlabel("Time Step")
        plt.ylabel("Sate 1")
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.plot(states_env[:, 2], label="Env Rewards", color="blue")
        plt.plot(states_model[:, 2], label="Model Rewards", color="red", linestyle="--")
        plt.title("State Comparison - State 2")
        plt.xlabel("Time Step")
        plt.ylabel("State 2")
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.plot(states_env[:, 3], label="Env Rewards", color="blue")
        plt.plot(states_model[:, 3], label="Model Rewards", color="red", linestyle="--")
        plt.title("State Comparison - State 3")
        plt.xlabel("Time Step")
        plt.ylabel("State 3")
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.plot(states_env[:, 4], label="Env Rewards", color="blue")
        plt.plot(states_model[:, 4], label="Model Rewards", color="red", linestyle="--")
        plt.title("State Comparison - State 4")
        plt.xlabel("Time Step")
        plt.ylabel("State 4")
        plt.legend()
        plt.subplot(2, 3, 6)
        plt.plot(rewards_env, label="Env Rewards", color="blue")
        plt.plot(rewards_model, label="Model Rewards", color="red", linestyle="--")
        plt.title("Reward Comparison")
        plt.xlabel("Time Step")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("trajectory_comparison.png")

    compare_trajectories(env, carry[3], carry[0], num_timesteps=15)


if __name__ == "__main__":
    test_train_model()
