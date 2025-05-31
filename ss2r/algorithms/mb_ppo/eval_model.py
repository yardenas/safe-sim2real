import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from brax import envs
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint

from ss2r.algorithms.mb_ppo import networks as mb_ppo_networks
from ss2r.algorithms.mb_ppo.model_env import ModelBasedEnv
from ss2r.benchmark_suites.brax.cartpole import cartpole


def get_state_path():
    return "./wandb_checkpoints"


def get_wandb_checkpoint(run_id):
    api = wandb.Api()
    artifact = api.artifact(f"ss2r/checkpoint:{run_id}")
    download_dir = artifact.download(f"{get_state_path()}/{run_id}")
    return download_dir


def plot(y_true, y_pred, savename):
    t = np.arange(y_true.shape[0])
    plt.figure(figsize=(6, 4), dpi=600)
    plt.plot(t, y_pred, "r", label="prediction", linewidth=1.0)
    plt.plot(t, y_true, "c", label="ground truth", linewidth=1.0)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savename, bbox_inches="tight")
    plt.show(block=False)
    plt.close()


def plot_observations(obs_true, obs_pred, savename_prefix):
    """Plot each observation dimension separately."""
    t = np.arange(obs_true.shape[0])
    obs_dim = obs_true.shape[1]

    for dim in range(obs_dim):
        plt.figure(figsize=(6, 4), dpi=600)
        plt.plot(t, obs_pred[:, dim], "r", label=f"prediction dim {dim}", linewidth=1.0)
        plt.plot(
            t, obs_true[:, dim], "c", label=f"ground truth dim {dim}", linewidth=1.0
        )
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{savename_prefix}_dim{dim}.png", bbox_inches="tight")
        plt.close()


def run_policy(env, policy_fn, steps=15, key=jax.random.PRNGKey(0)):
    state = jax.jit(env.reset)(key)
    actions = []
    rewards = []
    obs = []
    for _ in range(steps):
        key, subkey = jax.random.split(key)
        action, _ = policy_fn(state.obs, subkey)
        state = jax.jit(env.step)(state, action)
        actions.append(action)
        rewards.append(state.reward)
        obs.append(state.obs)
    return jnp.stack(rewards), jnp.stack(actions), jnp.stack(obs)


def evaluate_model(model_apply_fn, obs, actions, key, horizon=15):
    pred_rewards = []
    pred_obs = []
    current_state = obs[0]
    current_state = envs.State(
        pipeline_state=None,
        obs=current_state,
        reward=jnp.zeros((1,)),
        done=jnp.zeros((1,)),
        info={
            "cumulative_cost": jnp.zeros((1,)),
            "truncation": jnp.zeros((1,)),
            "cost": jnp.zeros((1,)),
        },
    )
    for t in range(horizon):
        action = actions[t]
        next_state = jax.jit(model_apply_fn)(current_state, action)
        pred_rewards.append(next_state.reward)
        pred_obs.append(next_state.obs)
        current_state = next_state
    pred_rewards = jnp.stack(pred_rewards)  # [horizon, obs_dim]
    pred_obs = jnp.stack(pred_obs)  # [horizon, obs_dim]
    return pred_rewards, pred_obs


def main(run_id):
    checkpoint_path = get_wandb_checkpoint(run_id)

    # === Load model ===
    params = checkpoint.load(checkpoint_path)
    (
        normalizer_params,
        policy_params,
        value_params,
        cost_value_params,
        model_params,
    ) = params
    # === Brax environment and policy ===
    env = cartpole.Cartpole(backend="generalized", swingup=False, sparse=False)
    obs_size = env.observation_size
    action_size = env.action_size
    normalize_fn = running_statistics.normalize
    denormalize_fn = running_statistics.denormalize
    ppo_network = mb_ppo_networks.make_mb_ppo_networks(
        obs_size,
        action_size,
        normalize_fn,
        denormalize_fn,
        model_hidden_layer_sizes=(512, 512),
        use_bro=False,
    )
    make_policy = mb_ppo_networks.make_inference_fn(ppo_network)
    policy = make_policy((normalizer_params, policy_params), deterministic=True)
    model = ModelBasedEnv(
        model_network=ppo_network.model_network,
        model_params=model_params,
        normalizer_params=normalizer_params,
        action_size=env.action_size,
        safety_budget=float("inf"),
        ensemble_selection="mean",  # Choose from 'random', 'mean', or 'pessimistic'
        observation_size=env.observation_size,
    )
    # === Run policy to collect a trajectory ===
    key = jax.random.PRNGKey(42)
    horizon = 25
    rewards, actions, obs = run_policy(env, policy, steps=horizon, key=key)
    # === Evaluate model over final `horizon` steps ===
    pred_rewards, pred_obs = evaluate_model(
        model.step, obs, actions, key, horizon=horizon
    )
    plot(rewards, pred_rewards, "rewards.png")
    plot_observations(obs, pred_obs, "observations")


if __name__ == "__main__":
    run_id = "k7r264l2"  # <-- Replace with your actual run ID
    main(run_id)
