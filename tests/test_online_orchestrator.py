import threading
import time

import cloudpickle as pickle
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
            transitions = [
                Transition(
                    observation=np.random.rand(4),
                    action=np.random.rand(2),
                    reward=np.random.rand(1),
                    discount=np.array([0.99]),
                    next_observation=np.random.rand(4),
                )
                for _ in range(num_steps)
            ]
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


def dummy_translate_policy_to_binary_fn(policy):
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
        open_reverse_tunnel=False,
    )
    policy = DummyPolicy(param=123)
    policy_bytes = dummy_translate_policy_to_binary_fn(policy)
    response = orchestrator._send_request(policy_bytes)
    assert isinstance(response, tuple)
    assert len(response) == 2
    success, transitions = response
    assert success is True
    assert isinstance(transitions, list)
    assert len(transitions) == 100
    assert all(isinstance(t, Transition) for t in transitions)
