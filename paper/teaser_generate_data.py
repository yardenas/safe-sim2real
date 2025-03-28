# %%
import argparse
import functools
import pickle

import jax
import jax.nn as jnn
import numpy as np
from brax.training.acme import running_statistics
from hydra import compose, initialize

import ss2r.algorithms.sac.networks as sac_networks
from ss2r import benchmark_suites
from ss2r.algorithms.sac.wrappers import PTSD, ModelDisagreement
from ss2r.rl.utils import rollout

# %%
parser = argparse.ArgumentParser(description="Load and play policy")
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=False,
    default=None,
    help="Path to the checkpoint file",
)

args = parser.parse_args()
# %%


def make_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "writers=[stderr]",
                "+experiment=go1_sim_to_real",
                "training.num_envs=96",
                # Set more agressive commands to make it fall
                "environment.task_params.command_config.a=[1.5, 1.2, 2.5]",
                "environment.task_params.command_config.b=[0.99, 0.99, 0.99]",
            ]
            + additional_overrides,
        )
        return cfg


cfg = make_config()


def wrap_env(env):
    key = jax.random.PRNGKey(cfg.training.seed)
    env = PTSD(
        env,
        benchmark_suites.prepare_randomization_fn(
            key,
            cfg.agent.propagation.num_envs,
            cfg.environment.train_params,
            cfg.environment.task_name,
        ),
        cfg.agent.propagation.num_envs,
    )
    env = ModelDisagreement(env)
    return env


train_env, eval_env = benchmark_suites.make(cfg, wrap_env)


# %%
def identity_observation_preprocessor(observation, preprocessor_params):
    del preprocessor_params
    return observation


def load_disk_policy():
    activation = getattr(jnn, cfg.agent.activation)
    ckpt_path = args.ckpt_path
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        activation=activation,
        value_obs_key="state"
        if not cfg.training.value_privileged
        else "privileged_state",
        policy_obs_key="state",
        preprocess_observations_fn=running_statistics.normalize,
    )
    sac_network = network_factory(train_env.observation_size, train_env.action_size)
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    return inference_fn


inference_fn = load_disk_policy()

# %%

rng = jax.random.PRNGKey(0)


@jax.jit
def collect_episode(key):
    keys = jax.random.split(key, cfg.training.num_envs)
    state = train_env.reset(keys)
    _, trajectory = rollout(train_env, inference_fn, 1000, key, state)
    return trajectory


terminal = False
while not terminal:
    print("Collecting trajectory...")
    rng, rng_ = jax.random.split(rng)
    trajectory = collect_episode(rng_)
    truncation = trajectory.info["truncation"]
    terminated = trajectory.done.astype(bool) & (~truncation.astype(bool))
    terminal = terminated.any()

trajectory = jax.tree_map(
    lambda x: jax.device_put(x, jax.devices("cpu")[0]), trajectory
)
only_terminated_trajectories = jax.tree_map(
    lambda x: x[:, terminated.any(axis=0)],
    trajectory,
)


def split_trajectories(trajectory):
    done_array = np.array(trajectory.done)
    trajectories = []
    for i in range(done_array.shape[1]):
        indices = np.where(done_array[:, i] == 1)[0]
        # Add the first index (0) at the beginning to capture the initial segment
        split_indices = np.concatenate(([0], indices + 1))
        for j in range(len(split_indices) - 1):
            # Extract sub-trajectory
            start, end = split_indices[j], split_indices[j + 1]
            sub_trajectory = jax.tree_map(lambda x: x[start:end, i], trajectory)
            trajectories.append(sub_trajectory)
        # Add the remaining part of the trajectory after the last done=True, if any
        if split_indices[-1] < done_array.shape[0]:
            sub_trajectory = jax.tree_map(
                lambda x: x[split_indices[-1] :, i], trajectory
            )
            trajectories.append(sub_trajectory)
    return trajectories


flattened_trajectories = split_trajectories(trajectory)
with open("trajectory.pkl", "wb") as f:
    pickle.dump(flattened_trajectories, f)
print("Trajectory saved.")
print(trajectory.info["disagreement"])
