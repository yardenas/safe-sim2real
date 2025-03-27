# %%
import argparse
import functools
import pickle

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.acting import generate_unroll
from hydra import compose, initialize

import ss2r.algorithms.sac.networks as sac_networks
from ss2r import benchmark_suites
from ss2r.algorithms.sac.wrappers import PTSD, ModelDisagreement

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
                "training.num_envs=8192",
                # Set more agressive commands to make it fall
                "environment.task_params.command_config.a=[5., 5.2, 5.2]",
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
    state, trajectory = generate_unroll(
        train_env,
        state,
        inference_fn,
        key,
        unroll_length=1000,
        extra_fields=("truncation", "disagreement"),
    )
    return state, trajectory


terminal = False
while not terminal:
    print("Collecting trajectory...")
    rng, rng_ = jax.random.split(rng)
    state, trajectory = collect_episode(rng_)
    truncation = state.info["truncation"]
    terminal = state.done.astype(bool) & (~truncation.astype(bool))
    trajectory = jax.tree_map(
        lambda x: jnp.take(x, jnp.nonzero(terminal)[0], axis=0), trajectory
    )
    terminal = terminal.any()  # type: ignore
    # Save trajectory using pickle
    with open("trajectory.pkl", "wb") as f:
        pickle.dump(trajectory, f)

    print("Trajectory saved.")

print(trajectory.extras["state_extras"]["disagreement"])
