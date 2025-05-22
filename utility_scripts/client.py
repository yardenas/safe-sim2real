import functools

import cloudpickle as pickle
import jax
import tensorflow as tf
import tf2onnx
import zmq

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac.go1_sac_to_onnx import make_policy_network, transfer_weights


def load_policy_from_init(obs_size, act_size):
    # Manually set the config parameters
    activation = jax.nn.relu  # Replace with your desired activation, e.g., jnn.tanh
    value_hidden_layer_sizes = [256, 256]
    policy_hidden_layer_sizes = [256, 256]
    value_obs_key = "state"  # Or "privileged_state", depending on your use case
    policy_obs_key = "state"
    # Build the network
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        value_obs_key=value_obs_key,
        policy_obs_key=policy_obs_key,
    )
    # Create networks
    sac_network = network_factory(obs_size, act_size)
    # Initialize parameters instead of loading from checkpoint
    rng = jax.random.PRNGKey(0)  # You may want to pass in or randomize the seed
    params = sac_network.policy_network.init(rng)
    # Inference function
    make_inference_fn = sac_networks.make_inference_fn(sac_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    return inference_fn, params


def convert_to_onnx(params, act_size, obs_size):
    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=None,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.swish,
    )
    example_input = tf.zeros((1, obs_size))
    tf_policy_network(example_input)
    # Transfer JAX weights to TF model
    transfer_weights(params["params"], tf_policy_network)
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


def send_request(address="tcp://localhost:5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)
    _, params = load_policy_from_init(obs_size=48, act_size=12)
    model_proto = convert_to_onnx(params, act_size=12, obs_size=48)
    # Serialize policy and number of steps
    message = pickle.dumps((model_proto.SerializeToString(), 1000))
    print("Sending request to server...")
    socket.send(message)
    # Receive and deserialize response
    response = socket.recv()
    print("Received response from server...")
    transitions = pickle.loads(response)
    observation, action, reward, discount, next_observation, info = transitions
    for i, t in enumerate(transitions):
        print(f"\nTransition {i + 1}:")
        print(f"  Observation: {observation}")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Discount: {discount}")
        print(f"  Next Observation: {next_observation}")

    socket.close()
    context.term()


if __name__ == "__main__":
    send_request()
