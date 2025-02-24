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
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import locomotion
from mujoco_playground.config import locomotion_params
from tensorflow.keras import layers  # type: ignore

# %%
parser = argparse.ArgumentParser(description="Convert Flax model to ONNX")
parser.add_argument(
    "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
)
parser.add_argument(
    "--normalize_obs", action="store_true", help="Use observation normalization"
)
args = parser.parse_args()

# %%
env_name = "Go1JoystickFlatTerrain"
ppo_params = locomotion_params.brax_ppo_config(env_name)


def identity_observation_preprocessor(observation, preprocessor_params):
    del preprocessor_params
    return observation


network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.network_factory,
    preprocess_observations_fn=running_statistics.normalize
    if args.normalize_obs
    else identity_observation_preprocessor,
)

# %%
env_cfg = locomotion.get_default_config(env_name)
env = locomotion.load(env_name, config=env_cfg)
obs_size = env.observation_size
act_size = env.action_size
print(obs_size, act_size)

# %%
ppo_network = network_factory(obs_size, act_size)
ckpt_path = args.ckpt_path

# %%
with open(ckpt_path, "rb") as f:
    params = pickle.load(f)
print(params.keys())

# %%
output_path = "bh_policy.onnx"
params = (params["normalizer_params"], params["policy_params"])
make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
inference_fn = make_inference_fn(params, deterministic=True)


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
        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(layer_sizes):
            dense_layer = layers.Dense(
                size,
                activation=activation,
                kernel_initializer=kernel_init,
                name=f"hidden_{i}",
                use_bias=bias,
            )
            self.mlp_block.add(dense_layer)
            if layer_norm:
                self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
        if not activate_final and self.mlp_block.layers:
            self.mlp_block.layers[-1].activation = None

    def call(self, inputs):
        # TODO (yarden): double check this.
        return tf.tanh(self.mlp_block(inputs))


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
mean = params[0].mean["state"]
std = params[0].std["state"]
mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))
tf_policy_network = make_policy_network(
    param_size=act_size * 2,
    mean_std=mean_std,
    hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
    activation=tf.nn.swish,
)

# %%

example_input = tf.zeros((1, obs_size["state"][0]))
example_output = tf_policy_network(example_input)
print(example_output.shape)


# %%
def transfer_weights(jax_params, tf_model):
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            kernel = layer_params["kernel"]
            bias = layer_params["bias"]
            tf_layer.set_weights([kernel, bias])
        except ValueError:
            print(f"Layer {layer_name} not found.")
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
print(jax_pred)

# %%

plt.plot(onxx_pred, label="onnx")
plt.plot(tensorflow_pred, label="tensorflow")
plt.plot(jax_pred, label="jax")
plt.legend()
plt.show()
