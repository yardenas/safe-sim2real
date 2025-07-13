from types import SimpleNamespace

import jax
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac.franka_sac_to_onnx import make_policy_network
from ss2r.algorithms.sac.vision_networks import make_policy_vision_network


def test_policy_to_onnx_export():
    # Define dummy config
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            policy_hidden_layer_sizes=(256, 256), encoder_hidden_dim=128
        )
    )

    act_size = 4
    obs_shape = (64, 64, 3)  # shape for each image input
    batch_size = 1
    # Dummy observation with a single pixel input
    obs = {"pixels/view_0": np.ones((batch_size,) + obs_shape, dtype=tf.float32)}

    policy_network = make_policy_vision_network(
        observation_size={"pixels/view_0": obs_shape},
        output_size=act_size,
        hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
        activation=tf.nn.swish,
        tanh=True,
    )
    params = sac_networks.policy_network.init(jax.random.PRNGKey(0), policy_network)
    make_inference_fn = sac_networks.make_inference_fn(policy_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    # Create model
    tf_model = make_policy_network(
        output_size=act_size,
        hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
        activation=tf.nn.swish,
        tanh=True,
    )

    # Run a forward pass
    tensorflow_pred = tf_model(obs).numpy()[0]
    print("TF prediction:", tensorflow_pred)
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        tf_model,
        input_signature=[
            {
                "pixels/view_0": tf.TensorSpec(
                    [1, 64, 64, 3], tf.float32, name="pixels/view_0"
                )
            }
        ],
        output_path="temp.onnx",
        opset=11,
    )
    # Load ONNX model
    sess = rt.InferenceSession("temp.onnx", providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(["continuous_actions"], obs)[0][0]
    print("ONNX prediction:", onnx_pred)
    jax_pred, _ = inference_fn(obs, jax.random.PRNGKey(0))
    print("JAX prediction:", jax_pred)
    # Plot comparison
    plt.plot(onnx_pred[:act_size], label="onnx")
    plt.plot(tensorflow_pred[:act_size], label="tensorflow")
    plt.plot(jax_pred, label="jax")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_policy_to_onnx_export()
