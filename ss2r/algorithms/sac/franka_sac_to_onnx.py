import numpy as np

from ss2r.algorithms.sac.go1_sac_to_onnx import MLP

try:
    import tensorflow as tf
    import tf2onnx
    from tensorflow.keras import layers  # type: ignore
except ImportError:
    tf = None
    layers = None
    import logging

    logging.warning("TensorFlow is not installed. Skipping conversion to ONNX.")


class Encoder(tf.keras.Model):
    def __init__(
        self,
        features=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        padding="same",
        name="SharedEncoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.features = features
        self.strides = strides
        self.padding = padding
        self.conv_block = tf.keras.Sequential(name="CNN_0")

        for i, (f, s) in enumerate(zip(features, strides)):
            layer = layers.Conv2D(
                filters=f,
                kernel_size=3,
                strides=s,
                padding=padding,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=tf.sqrt(2.0)),
                activation="relu",
                name=f"Conv_{i}",
            )
            self.conv_block.add(layer)

    def call(self, data):
        cnn_outs = []
        for key, x in data.items():
            if key.startswith("pixels/"):
                x = self.conv_block(x)
                if len(x.shape) == 4:
                    x = tf.reshape(x, [x.shape[0], -1])
                else:
                    x = tf.reshape(x, [-1])
                cnn_outs.append(x)
        return tf.concat(cnn_outs, axis=-1)


class Policy(tf.keras.Model):
    def __init__(
        self,
        action_size,
        encoder_hidden_dim=50,
        hidden_layer_sizes=(256, 256),
        activation=tf.nn.swish,
        tanh=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = Encoder()
        self.encoder_dense = layers.Dense(encoder_hidden_dim, name="encoder_dense")
        self.encoder_norm = layers.LayerNormalization(name="encoder_norm")
        self.use_tanh = tanh
        self.tanh = tf.nn.tanh if tanh else lambda x: x
        self.mlp = MLP(  # Assuming your previously defined MLP class is available
            layer_sizes=list(hidden_layer_sizes) + [action_size * 2],
            activation=activation,
        )
        self.submodules = [self.encoder, self.encoder_dense, self.mlp]

    def call(self, obs):
        hidden = self.encoder(obs)
        hidden = self.encoder_dense(hidden)
        hidden = self.encoder_norm(hidden)
        hidden = self.tanh(hidden)
        return self.mlp(hidden)


def make_policy_network(
    action_size,
    hidden_layer_sizes=(256, 256),
    activation=tf.nn.swish,
    encoder_hidden_dim: int = 50,
    tanh: bool = True,
):
    return Policy(
        action_size=action_size,
        encoder_hidden_dim=encoder_hidden_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        tanh=tanh,
    )


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
    tf_inner_module_names = {
        "mlp": "MLP_0",
        "SharedEncoder": "CNN_0",
        "encoder_dense": "Dense_0",
    }
    tf_to_jax_module = {
        "mlp": "MLP_0",
        "SharedEncoder": "SharedEncoder",
        "encoder_dense": "Dense_0",
    }
    for module_name, inner_module_name in tf_inner_module_names.items():
        for layer_name, layer_params in jax_params[
            tf_to_jax_module[module_name]
        ].items():
            try:
                if inner_module_name == "Dense_0":
                    tf_layer = tf_model.get_layer(module_name)
                    layer_params = jax_params[tf_to_jax_module[module_name]]
                else:
                    tf_layer = (
                        tf_model.get_layer(module_name)
                        .get_layer(inner_module_name)
                        .get_layer(name=layer_name)
                    )
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
                kernel = np.array(layer_params["kernel"])
                bias = np.array(layer_params["bias"])
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")


def convert_policy_to_onnx(params, cfg, act_size, obs_size):
    """
    Converts a JAX policy to ONNX format using a TensorFlow intermediate model.

    Args:
        jax_params (dict): Parameters of the JAX policy model.
        policy_cfg (OmegaConf): Configuration object with network architecture info.
        act_size (int): Size of the action space.
        obs_shape (int): Flattened size of the observation input.
        output_path (str): Path to save the ONNX model.
        mean_std (Tuple[tf.Tensor, tf.Tensor] or None): Optional normalization tensors.

    Returns:
        onnx_prediction (np.ndarray): Output from the ONNX model for sanity check.
        onnx_model (InferenceSession): Loaded ONNX runtime model.
    """
    # Define the TF policy network
    tf_policy_network = make_policy_network(
        action_size=act_size,
        hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
        encoder_hidden_dim=cfg.agent.encoder_hidden_dim,
    )
    # Transfer JAX weights to TF model
    transfer_weights(params[1]["params"], tf_policy_network)
    # Export to ONNX
    tf_policy_network.output_names = ["continuous_actions"]
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network,
        input_signature=[
            tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
        ],
        opset=11,
    )
    return model_proto


def make_franka_policy(make_policy_fn, params, cfg):
    del make_policy_fn
    proto_model = convert_policy_to_onnx(params, cfg, 4, (64, 64, 3))
    return proto_model.SerializeToString()
