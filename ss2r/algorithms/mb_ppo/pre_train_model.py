import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from brax.training import gradients
from brax.training.acme import running_statistics
from brax.training.types import Transition


def create_env_data(env, key, num_samples, batch_size):
    """Create environment data by sampling states uniformly over the domain."""
    # Compute the rollout length and number of rollouts
    rollout_length = 100
    num_rollouts = num_samples // (batch_size * rollout_length)

    # Split the random key
    key, subkey = jax.random.split(key)

    def rollout_step(carry, _):
        state, key = carry
        key, action_key = jax.random.split(key)

        # Get batch size from state
        batch_dim = state.obs.shape[0]

        # Sample random actions - with correct batch dimension
        action = jax.random.uniform(
            action_key, shape=(batch_dim, env.action_size), minval=-1.0, maxval=1.0
        )

        # Step the environment
        next_state = env.step(state, action)

        # Extract observations, rewards, and costs
        obs = state.obs
        next_obs = next_state.obs
        reward = next_state.reward
        cost = next_state.info.get("cost", jnp.zeros_like(reward))

        transitions = (obs, action, next_obs, reward, cost)
        return (next_state, key), transitions

    def single_traj(key):
        # Split keys for initial state sampling and rollout
        key, reset_key, rollout_key = jax.random.split(key, 3)
        keys = jax.random.split(reset_key, batch_size)
        reset_key = keys[:batch_size].reshape(batch_size, -1)

        # First get a reference state
        initial_state = env.reset(reset_key)

        # Perform rollout
        (_, _), data = jax.lax.scan(
            rollout_step, (initial_state, rollout_key), None, length=rollout_length
        )

        return data

    # Use scan to execute trajectories sequentially
    def scan_body(key, _):
        key, traj_key = jax.random.split(key)
        data = single_traj(traj_key)
        return key, data

    _, data = jax.lax.scan(scan_body, key, None, length=num_rollouts)
    # Flatten the batch dimension
    data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[3:]), data)
    # Convert data to arrays
    obs, actions, next_obs, rewards, costs = data

    # Create transitions object
    transitions = create_transitions(obs, actions, next_obs, rewards, costs)

    return transitions


def create_transitions(obs, actions, next_obs, rewards, costs):
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


def pre_train_model(
    params,
    model_network,
    normalizer_params,
    model_optimizer,
    optimizer_state,
    env,
    model_loss,
    num_samples=1e5,
    batch_size=1024,
    epochs=100,
):
    key = jax.random.PRNGKey(0)
    gen_key, key_sgd = jax.random.split(key, 2)

    # Generate synthetic training data
    # obs, actions, next_obs, rewards, costs = create_synthetic_training_data(gen_key, obs_size, action_size, num_samples)
    transitions = create_env_data(env, gen_key, num_samples, batch_size)

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

    def eval_model_mse(params, normalizer_params, transitions):
        """Evaluate the model using mean squared error."""
        model_apply = model_network.apply
        (
            (diff_next_obs_pred, reward_pred, cost_pred),
            (next_obs_std, reward_std, cost_std),
        ) = model_apply(
            normalizer_params, params, transitions.observation, transitions.action
        )

        next_obs_pred = transitions.observation + jnp.mean(
            diff_next_obs_pred, axis=0
        )  # [batch, state_dim]
        reward_pred = jnp.mean(reward_pred, axis=0)  # [batch, 1]
        cost_pred = jnp.mean(cost_pred, axis=0)  # [batch, 1]

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
        key_sgd, sample_key = jax.random.split(key_sgd, 2)
        # Sample a batch of transitions of size 100
        transitions_batch = jax.tree.map(
            lambda x: jax.random.permutation(sample_key, x)[:batch_size], transitions
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

        (params, optimizer_state, transitions, key_sgd), aux_hist = jax.lax.scan(
            sgd_step, (params, optimizer_state, transitions, subkey), None, length=100
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
    plt.savefig("pretrained_model_training_mse.png")

    def compare_trajectories(
        key,
        env,
        normalizer_params,
        params,
        model_network,
        num_timesteps=15,
        num_trajs=5,
    ):
        # Generate reset keys for each trajectory
        reset_key, rollout_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_trajs)

        # Reset environment with batch of keys
        initial_states_env = env.reset(reset_keys)
        initial_states_model = initial_states_env.obs

        def rollout_env(carry, _):
            state_env, key = carry
            key, action_key = jax.random.split(key)

            batch_dim = state_env.obs.shape[0]

            action = jax.random.uniform(
                action_key, shape=(batch_dim, env.action_size), minval=-1.0, maxval=1.0
            )

            next_state_env = env.step(state_env, action)

            return (next_state_env, key), (
                next_state_env.obs,
                next_state_env.reward,
                action,
            )

        (final_states_env, _), (obs_seqs, reward_seqs, action_seqs) = jax.lax.scan(
            rollout_env, (initial_states_env, rollout_key), None, length=num_timesteps
        )

        obs_env = jnp.concatenate([initial_states_env.obs[None, ...], obs_seqs], axis=0)
        rewards_env = jnp.concatenate([jnp.ones((1, num_trajs)), reward_seqs], axis=0)
        zero_actions = jnp.zeros((1, num_trajs, env.action_size))
        actions_env = jnp.concatenate([zero_actions, action_seqs], axis=0)

        def model_rollout_step(carry, action_t):
            states = carry

            def single_model_step(state, action):
                ((diff_next_state, reward, cost), _) = model_network.apply(
                    normalizer_params, params, state, action
                )
                return diff_next_state, reward, cost

            # Apply to each state-action pair in the batch
            diff_next_states, rewards, costs = jax.vmap(single_model_step)(
                states, action_t
            )

            # Take mean across ensemble dimension
            next_states = states + jnp.mean(
                diff_next_states, axis=0
            )  # [batch, state_dim]
            rewards = jnp.mean(rewards, axis=0)  # [batch, 1]
            costs = jnp.mean(costs, axis=0)  # [batch, 1]

            return next_states, (next_states, rewards, costs)

        model_actions = actions_env[1:]

        (
            next_states,
            (obs_model_seqs, reward_model_seqs, cost_model_seqs),
        ) = jax.lax.scan(
            model_rollout_step,
            initial_states_model,  # Initial carry has shape [batch, state_dim]
            model_actions,  # Actions have shape [time, batch, action_dim]
        )

        obs_model = jnp.concatenate(
            [initial_states_model[None, ...], obs_model_seqs], axis=0
        )
        rewards_model = jnp.concatenate(
            [jnp.ones((1, num_trajs)), reward_model_seqs], axis=0
        )

        plt.figure(figsize=(18, 10))
        for traj in range(num_trajs):
            for i in range(min(5, obs_env.shape[2])):  # up to 5 state dims
                plt.subplot(num_trajs, 6, traj * 6 + i + 1)
                plt.plot(obs_env[:, traj, i], label="Env", color="blue")
                plt.plot(
                    obs_model[:, traj, i], label="Model", color="red", linestyle="--"
                )
                plt.title(f"Traj {traj+1} - State {i}")
                if traj == 0:
                    plt.legend()
            plt.subplot(num_trajs, 6, traj * 6 + 6)
            plt.plot(rewards_env[:, traj], label="Env", color="blue")
            plt.plot(rewards_model[:, traj], label="Model", color="red", linestyle="--")
            plt.title(f"Traj {traj+1} - Reward")
            if traj == 0:
                plt.legend()
        plt.tight_layout()
        plt.savefig("pretrained_model_trajectory_comparison.png")

    compare_trajectories(
        key, env, carry[3], carry[0], model_network, num_timesteps=15, num_trajs=5
    )
    return carry[0], carry[3]  # Return final params and normalizer params
