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

from omegaconf import OmegaConf
import wandb
import jax
import jax.numpy as jp
import jax.nn as jnn
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from hydra import compose, initialize
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import locomotion
from mujoco_playground.config import locomotion_params
from tensorflow.keras import layers  # type: ignore

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
ppo_params = locomotion_params.brax_ppo_config(env_name)


def identity_observation_preprocessor(observation, preprocessor_params):
    del preprocessor_params
    return observation


def load_config():
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=["writers=[stderr]", "+experiment=go1_sim_to_real_ppo"],
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
    ppo_network = ppo_networks.make_ppo_networks(
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
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    return make_policy(policy_params, True), policy_params, cfg


def load_disk_policy():
    cfg = load_config()
    activation = getattr(jnn, cfg.agent.activation)
    ckpt_path = args.ckpt_path
    with open(ckpt_path, "rb") as f:
        params = pickle.load(f)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
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
    make_inference_fn = ppo_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)
    return inference_fn, params, cfg


# %%
class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
        mean_std=None,
    ):
        super().__init__()

        self.layer_sizes = layer_sizes
        self.activation = activation
        self.kernel_init = kernel_init
        self.activate_final = activate_final
        self.bias = bias
        self.layer_norm = layer_norm

        if mean_std is not None:
            self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
            self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        else:
            self.mean = None
            self.std = None

        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(self.layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=self.activation,
                kernel_initializer=self.kernel_init,
                name=f"hidden_{i}",
                use_bias=self.bias,
            )
            self.mlp_block.add(dense_layer)
            if self.layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        if not self.activate_final and self.mlp_block.layers:
            if (
                hasattr(self.mlp_block.layers[-1], "activation")
                and self.mlp_block.layers[-1].activation is not None
            ):
                self.mlp_block.layers[-1].activation = None

        self.submodules = [self.mlp_block]

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = inputs[0]
        if self.mean is not None and self.std is not None:
            print(self.mean.shape, self.std.shape)
            inputs = (inputs - self.mean) / self.std
        logits = self.mlp_block(inputs)
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)


# %%
def make_policy_network(
    param_size,
    mean_std,
    hidden_layer_sizes=[256, 256],
    activation=tf.nn.relu,
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
        mean_std=mean_std,
    )


# %%
if args.ckpt_path is not None:
    inference_fn, params, cfg = load_disk_policy()
else:
    inference_fn, params, cfg = fetch_wandb_policy(args.wandb_run_id)
mean = params[0].mean["state"]
std = params[0].std["state"]
mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
tf_policy_network = make_policy_network(
    param_size=act_size * 2,
    mean_std=mean_std,
    hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
    activation=tf.nn.swish,
)

# %%

example_input = tf.zeros((1, obs_size["state"][0]))
example_output = tf_policy_network(example_input)
print(example_output.shape)


# %%
def transfer_weights(jax_params, tf_model):
    """
    Transfer weights from a JAX parameter dictionary to the TensorFlow model.

    Parameters:
    - jax_params: dict
      Nested dictionary with structure {block_name: {layer_name: {params}}}.
      For example:
      {
        'CNN_0': {
          'Conv_0': {'kernel': np.ndarray},
          'Conv_1': {'kernel': np.ndarray},
          'Conv_2': {'kernel': np.ndarray},
        },
        'MLP_0': {
          'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
          'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
        }
      }

    - tf_model: tf.keras.Model
      An instance of the adapted VisionMLP model containing named submodules and layers.
    """
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
        except ValueError:
            print(f"Layer {layer_name} not found in TensorFlow model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            kernel = np.array(layer_params["kernel"])
            bias = np.array(layer_params["bias"])
            print(
                f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
            )
            tf_layer.set_weights([kernel, bias])
        else:
            print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")
    print("Weights transferred successfully.")


transfer_weights(params[1]["params"], tf_policy_network)

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
