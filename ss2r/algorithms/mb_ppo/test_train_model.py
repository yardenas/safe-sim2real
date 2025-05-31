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


def test_train_model(lr=1e-3, epochs=100, use_bro=False):
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
        use_bro=use_bro,
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
    model_optimizer = optax.adam(learning_rate=lr)
    optimizer_state = model_optimizer.init(params)

    # Update normalizer params
    normalizer_params = running_statistics.update(
        normalizer_params, transitions.observation
    )

    # Make loss
    model_loss, *_ = mb_ppo_losses.make_losses(
        ppo_network=ppo_network,
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
        params, optimizer_state, transitions, normalizer_params, key_sgd, num_epochs
    ):
        carry = (params, optimizer_state, transitions, normalizer_params, key_sgd)
        carry, mse_hist = jax.lax.scan(epoch_step, carry, None, length=num_epochs)
        return carry, mse_hist

    carry, mse_hist = train_model(
        params,
        optimizer_state,
        transitions,
        normalizer_params,
        key_sgd,
        num_epochs=epochs,
    )

    plt.plot(mse_hist[0], label="MSE Next Obs")
    plt.plot(mse_hist[1], label="MSE Reward")
    plt.plot(mse_hist[2], label="MSE Cost")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Model Training MSE")
    plt.legend()
    plt.savefig(f"model_training_mse_lr{lr}.png")

    def compare_trajectories(
        key, env, normalizer_params, params, num_timesteps=15, num_trajs=5
    ):
        keys = jax.random.split(key, num_trajs)
        initial_states_env = jax.vmap(env.reset)(keys)
        initial_states_model = initial_states_env.obs

        def rollout_env(carry, _):
            state_env, key = carry
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(
                subkey, (env.action_size,), minval=-1.0, maxval=1.0
            )
            next_state_env = env.step(state_env, action)
            return (next_state_env, key), (
                next_state_env.obs,
                next_state_env.reward,
                action,
            )

        def rollout_model(carry, action):
            state_model = carry
            (
                (next_state_model, reward_model, cost_model),
                _,
            ) = ppo_network.model_network.apply(
                normalizer_params, params, state_model, action
            )
            next_state_model = jnp.mean(next_state_model, axis=0)
            reward_model = jnp.mean(reward_model, axis=0)
            cost_model = jnp.mean(cost_model, axis=0)
            return next_state_model, (next_state_model, reward_model, cost_model)

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

        def single_model_traj(init_state_model, actions):
            _, (obs_seq, reward_seq, cost_seq) = jax.lax.scan(
                rollout_model, init_state_model, actions[1:]
            )
            obs_seq = jnp.concatenate([init_state_model[None, ...], obs_seq], axis=0)
            reward_seq = jnp.concatenate([jnp.ones((1,)), reward_seq], axis=0)
            cost_seq = jnp.concatenate([jnp.zeros((1,)), cost_seq], axis=0)
            return obs_seq, reward_seq, cost_seq

        obs_model, rewards_model, costs_model = jax.vmap(single_model_traj)(
            initial_states_model, actions_env
        )

        obs_env = jnp.array(obs_env)
        obs_model = jnp.array(obs_model)
        rewards_env = jnp.array(rewards_env)
        rewards_model = jnp.array(rewards_model)

        plt.figure(figsize=(18, 10))
        for traj in range(num_trajs):
            for i in range(min(5, obs_env.shape[2])):  # up to 5 state dims
                plt.subplot(num_trajs, 6, traj * 6 + i + 1)
                plt.plot(obs_env[traj, :, i], label="Env", color="blue")
                plt.plot(
                    obs_model[traj, :, i], label="Model", color="red", linestyle="--"
                )
                plt.title(f"Traj {traj+1} - State {i}")
                if traj == 0:
                    plt.legend()
            plt.subplot(num_trajs, 6, traj * 6 + 6)
            plt.plot(rewards_env[traj], label="Env", color="blue")
            plt.plot(rewards_model[traj], label="Model", color="red", linestyle="--")
            plt.title(f"Traj {traj+1} - Reward")
            if traj == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig(f"trajectory_comparison{lr}.png")

    compare_trajectories(key, env, carry[3], carry[0], num_timesteps=25, num_trajs=5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and test a model with specified learning rate"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for model training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--use_bro",
        type=bool,
        default=False,
        help="Use BRO (Bayesian Robust Optimization) in the model",
    )
    args = parser.parse_args()

    print(f"Training with learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Using BRO: {args.use_bro}")
    test_train_model(lr=args.lr, epochs=args.epochs, use_bro=args.use_bro)
