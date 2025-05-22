import cloudpickle as pickle
import numpy as np
import zmq

# Dummy "policy" object — could be anything serializable
dummy_policy = {"weights": np.random.rand(3, 3)}
num_steps = 5


def send_request(address="tcp://localhost:5555"):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(address)

    # Serialize policy and number of steps
    message = pickle.dumps((dummy_policy, num_steps))

    print("Sending request to server...")
    socket.send(message)

    # Receive and deserialize response
    response = socket.recv()
    success, done, transitions = pickle.loads(response)

    print(f"Success: {success}, Done: {done}")
    for i, t in enumerate(transitions):
        print(f"\nTransition {i + 1}:")
        print(f"  Observation: {t.observation}")
        print(f"  Action: {t.action}")
        print(f"  Reward: {t.reward}")
        print(f"  Discount: {t.discount}")
        print(f"  Next Observation: {t.next_observation}")

    socket.close()
    context.term()


if __name__ == "__main__":
    send_request()
