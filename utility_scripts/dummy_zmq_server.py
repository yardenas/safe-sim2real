import time

import cloudpickle as pickle
import jax
import numpy as np
import zmq

from ss2r.algorithms.sac import Transition


def dummy_server(address="tcp://*:5555", delay=0.1):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)
    print("Dummy server started...")
    try:
        while True:
            message = socket.recv()
            _, num_steps = pickle.loads(message)
            print("Received request!")
            transitions = Transition(
                observation={
                    "privileged_state": np.random.rand(num_steps, 123),
                    "state": np.random.rand(num_steps, 48),
                },
                action=np.random.rand(num_steps, 12),
                reward=np.random.rand(num_steps),
                discount=np.random.rand(num_steps),
                next_observation={
                    "privileged_state": np.random.rand(num_steps, 123),
                    "state": np.random.rand(num_steps, 48),
                },
                extras={
                    "policy_extras": {},
                    "state_extras": {
                        "cost": np.random.rand(num_steps),
                        "truncation": np.random.rand(num_steps),
                    },
                },
            )
            transitions = jax.tree.map(lambda x: x.astype(np.float32), transitions)
            response = (True, transitions)
            time.sleep(delay)
            socket.send(pickle.dumps(response))
    except zmq.ZMQError:
        pass
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    dummy_server()
