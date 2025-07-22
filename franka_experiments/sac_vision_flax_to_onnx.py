import jax
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from hydra import compose, initialize

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac import franka_sac_to_onnx
from ss2r.algorithms.sac.franka_sac_to_onnx import make_policy_network
from ss2r.algorithms.sac.vision_networks import make_sac_vision_networks


def get_cfg():
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="train_brax",
            overrides=[
                "writers=[stderr]",
                "+experiment=franka_online",
            ],
        )
        return cfg


def test_policy_to_onnx_export():
    # Define dummy config
    cfg = get_cfg()

    act_size = 4
    obs_shape = (64, 64, 3)  # shape for each image input
    batch_size = 1
    # Dummy observation with a single pixel input
    obs = {"pixels/view_0": np.ones((batch_size,) + obs_shape, dtype=np.float32)}
    sac_network = make_sac_vision_networks(
        observation_size={"pixels/view_0": obs_shape},
        action_size=act_size,
        policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
        activation=tf.nn.swish,
        tanh=True,
    )
    params = sac_network.policy_network.init(jax.random.PRNGKey(0))
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    dummy_normalizer_params = None
    inference_fn = make_inference_fn(
        (dummy_normalizer_params, params), deterministic=True
    )
    # Create model
    tf_model = make_policy_network(
        action_size=act_size,
        hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
        activation=tf.nn.swish,
        tanh=cfg.agent.tanh,
    )

    # Run a forward pass
    tf_model(obs).numpy()[0]
    franka_sac_to_onnx.transfer_weights(params["params"], tf_model)
    tensorflow_pred = tf_model(obs).numpy()[0]
    print("TF prediction:", tensorflow_pred)
    # Convert to ONNX
    tf_model.output_names = ["continuous_actions"]
    tf2onnx.convert.from_keras(
        tf_model,
        input_signature=[
            {
                "pixels/view_0": tf.TensorSpec(
                    [1, 64, 64, 3], tf.float32, name="pixels/view_0"
                )
            }
        ],
        output_path="model.onnx",
        opset=11,
    )
    # Load ONNX model
    sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    onnx_pred = sess.run(["continuous_actions"], obs)[0][0]
    print("ONNX prediction:", onnx_pred)
    jax_pred = inference_fn(obs, jax.random.PRNGKey(0))[0][0]
    print("JAX prediction:", jax_pred)
    # Plot comparison
    plt.plot(onnx_pred[:act_size], label="onnx")
    plt.plot(tensorflow_pred[:act_size], label="tensorflow")
    plt.plot(jax_pred, label="jax")
    plt.legend()
    plt.show()
    franka_sac_to_onnx.make_franka_policy(make_inference_fn, (None, params), cfg)


if __name__ == "__main__":
    test_policy_to_onnx_export()
