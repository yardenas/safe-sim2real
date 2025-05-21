# ruff: noqa
# type: ignore
# https://github.com/google-deepmind/mujoco_playground/blob/609168a02cc068193f0fdb4379c2030b388b073e/mujoco_playground/experimental/brax_network_to_onnx.ipynb#L364
# %%
import argparse
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# %%
import functools
import pickle

import jax
import jax.nn as jnn
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
import wandb
from brax.training.acme import running_statistics
from hydra import compose, initialize
from mujoco_playground import locomotion
from omegaconf import OmegaConf
from tensorflow.keras import layers  # type: ignore

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.common import go1_sac_to_onnx

# %%
parser = argparse.ArgumentParser(description="Convert Flax model to ONNX")
parser.add_argument(
    "--ckpt_path",
    type=str,
    required=False,
    default=None,
    help="Path to the checkpoint file",
)
parser.add_argument("--output_path", type=str, default="model.onnx", help="Output path")
parser.add_argument(
    "--normalize_obs", action="store_true", help="Use observation normalization"
)
parser.add_argument("--wandb_run_id", type=str, help="Weights & Biases Run ID")
args = parser.parse_args()

if args.ckpt_path and args.wandb_run_id:
    raise ValueError("Specify either --ckpt_path or --wandb_run_id, not both.")
# %%
env_name = "Go1JoystickFlatTerrain"


def identity_observation_preprocessor(observation, preprocessor_params):
    del preprocessor_params
    return observation


def load_config():
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=["writers=[stderr]", "+experiment=go1_simple"],
        )
        return cfg


# %%
env_cfg = locomotion.get_default_config(env_name)
env = locomotion.load(env_name, config=env_cfg)
obs_size = env.observation_size
act_size = env.action_size
print(obs_size, act_size)

# %%


def fetch_wandb_policy(run_id):
    api = wandb.Api()
    run = api.run(f"ss2r/{run_id}")
    policy_artifact = api.artifact(f"ss2r/policy:{run_id}")
    policy_dir = policy_artifact.download()
    path = os.path.join(policy_dir, "policy.pkl")
    with open(path, "rb") as f:
        policy_params = pickle.load(f)
    config = run.config
    activation = getattr(jnn, config["agent"]["activation"])
    normalize = (
        running_statistics.normalize
        if config["agent"]["normalize_observations"]
        else identity_observation_preprocessor
    )
    cfg = OmegaConf.create(config)
    sac_network = sac_networks.make_sac_networks(
        observation_size=obs_size,
        action_size=act_size,
        value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        activation=activation,
        value_obs_key="state"
        if not cfg.training.value_privileged
        else "privileged_state",
        policy_obs_key="state",
        preprocess_observations_fn=normalize,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    return make_policy(policy_params, True), policy_params, cfg


def load_disk_policy():
    cfg = load_config()
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
        preprocess_observations_fn=running_statistics.normalize
        if args.normalize_obs
        else identity_observation_preprocessor,
    )
    sac_network = network_factory(obs_size, act_size)
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    return inference_fn, params, cfg


output_path = args.output_path
if args.ckpt_path is not None:
    inference_fn, params, cfg = load_disk_policy()
else:
    inference_fn, params, cfg = fetch_wandb_policy(args.wandb_run_id)

example_input = tf.zeros((1, obs_size["state"][0]))
# %%


# %%


# %%
if args.ckpt_path is not None:
    inference_fn, params, cfg = load_disk_policy()
else:
    inference_fn, params, cfg = fetch_wandb_policy(args.wandb_run_id)
mean = params[0].mean["state"]
std = params[0].std["state"]
mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
tf_policy_network = go1_sac_to_onnx.make_policy_network(
    param_size=act_size * 2,
    mean_std=mean_std,
    hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
    activation=tf.nn.swish,
)

# %%

example_input = tf.zeros((1, obs_size["state"][0]))
example_output = tf_policy_network(example_input)
print(example_output.shape)
go1_sac_to_onnx.transfer_weights(params[1]["params"], tf_policy_network)

# %%
test_input = [np.ones((1, obs_size["state"][0]), dtype=np.float32)]
tensorflow_pred = tf_policy_network(test_input)[0]
print(f"Tensorflow prediction: {tensorflow_pred}")

# %%
test_input = [np.ones((1, obs_size["state"][0]), dtype=np.float32)]
tensorflow_pred = tf_policy_network(test_input)[0]
print(f"Tensorflow prediction: {tensorflow_pred}")

# %%
tf_policy_network.output_names = ["continuous_actions"]
model_proto, _ = tf2onnx.convert.from_keras(
    tf_policy_network,
    input_signature=[
        tf.TensorSpec(shape=(1, obs_size["state"][0]), dtype=tf.float32, name="obs")
    ],
    opset=11,
    output_path=output_path,
)

# %%
output_names = ["continuous_actions"]
providers = ["CPUExecutionProvider"]
m = rt.InferenceSession(output_path, providers=providers)
onxx_input = {"obs": np.ones((1, obs_size["state"][0]), dtype=np.float32)}
onxx_pred = m.run(output_names, onxx_input)[0][0]
print("ONNX prediction:", onxx_pred)

# %%
test_input = {
    "state": jp.ones(obs_size["state"]),
    "privileged_state": jp.zeros(obs_size["privileged_state"]),
}
jax_pred, _ = inference_fn(test_input, jax.random.PRNGKey(0))
print("JAX prediction:", jax_pred)

# %%

plt.plot(onxx_pred[:act_size], label="onnx")
plt.plot(tensorflow_pred[:act_size], label="tensorflow")
plt.plot(jax_pred, label="jax")
plt.legend()
plt.show()
