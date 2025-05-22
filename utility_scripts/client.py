import cloudpickle as pickle
import numpy as np
import zmq

# Dummy "policy" object — could be anything serializable
dummy_policy = {"weights": np.random.rand(3, 3)}
num_steps = 1000


def send_request(address="tcp://localhost:5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)
    # Serialize policy and number of steps
    message = pickle.dumps((pickle.dumps(dummy_policy), num_steps))
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
