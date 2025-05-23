import threading
import time

import cloudpickle as pickle
import jax
import numpy as np
import pytest
import zmq
from brax.training.types import Transition

from ss2r.rl.online import OnlineEpisodeOrchestrator


# Dummy ZMQ server
def dummy_server(address="tcp://*:5555", delay=0.1):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)
    try:
        while True:
            message = socket.recv()
            _, num_steps = pickle.loads(message)
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


# Dummy policy and translation function
class DummyPolicy:
    def __init__(self, param):
        self.param = param


def dummy_translate_policy_to_binary_fn(policy_fn, policy):
    return pickle.dumps(policy)


# Fixture for server setup
@pytest.fixture(scope="module", autouse=True)
def start_dummy_server():
    thread = threading.Thread(target=dummy_server, daemon=True)
    thread.start()
    time.sleep(0.5)  # Ensure server is ready
    yield
    # Server thread exits automatically due to daemon=True


def test_send_request():
    orchestrator = OnlineEpisodeOrchestrator(
        translate_policy_to_binary_fn=dummy_translate_policy_to_binary_fn,
        num_steps=100,
        address="tcp://localhost:5555",
    )
    policy = DummyPolicy(param=123)
    transitions = orchestrator._send_request(lambda x: x, policy)
    assert transitions.observation["privileged_state"].shape == (100, 123)
    assert transitions.observation["state"].shape == (100, 48)
