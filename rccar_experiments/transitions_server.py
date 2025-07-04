import logging
import time

import cloudpickle as pickle
import numpy as np
import zmq

_LOG = logging.getLogger(__name__)


class TransitionsServer:
    def __init__(self, experiment_driver, safe_mode=False, address="tcp://*:5555"):
        self.experiment_driver = experiment_driver
        self.address = address
        self.safe_mode = safe_mode

    def loop(self):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REP) as socket:
                socket.bind(self.address)
                while True:
                    message = socket.recv()
                    policy_bytes, num_steps = pickle.loads(message)
                    if num_steps < self.experiment_driver.episode_length:
                        _LOG.error("Invalid num_steps: {}".format(num_steps))
                    trials = self.run(policy_bytes, num_steps)
                    if trials is None:
                        continue
                    socket.send(pickle.dumps(trials))

    def run(self, policy_bytes, num_steps):
        trials = []
        num_transitions = 0
        while num_transitions < num_steps:
            trial = self.do_trial(policy_bytes)
            new_num_transitions = len(trial)
            if num_transitions + new_num_transitions > num_steps:
                trial = trial[: num_steps - num_transitions]
                trial[-1].extras["state_extras"]["truncation"] = True
                _LOG.info("Truncating trajectory")
            num_transitions += len(trial)
            trials.append(trial)
            _LOG.info("Completed trial")
        transitions = flatten_trajectories(trials)
        assert (
            len(transitions[2]) == num_steps
        ), f"Expected {num_steps} transitions, got {len(transitions)}"
        return transitions

    def do_trial(self, policy_bytes):
        _LOG.info("Starting sampling")
        if self.safe_mode:
            while True:
                answer = input("Press Y/y when ready to collect trajectory\n")
                if not (answer == "Y" or answer == "y"):
                    _LOG.info("Skipping trajectory")
                    continue
                else:
                    break
        else:
            while not self.experiment_driver.robot_ok:
                _LOG.info("Waiting the robot to be ready...")
                time.sleep(2.5)
        policy_fn = self.parse_policy(policy_bytes)
        trajectory = self.experiment_driver.sample_trajectory(policy_fn)
        _LOG.info("Sampling finished")
        return trajectory

    def parse_policy(self, policy_bytes):
        policy_params = pickle.loads(policy_bytes)
        return self.experiment_driver.rollout_policy_fn(policy_params, True)


def flatten_trajectories(trajectories):
    if isinstance(trajectories[0][0].observation, dict):
        observations = {
            key: np.array(
                [t.observation[key] for traj in trajectories for t in traj],
                dtype=np.float32,
            )
            for key in trajectories[0][0].observation
        }
        next_observations = {
            key: np.array(
                [t.next_observation[key] for traj in trajectories for t in traj],
                dtype=np.float32,
            )
            for key in trajectories[0][0].next_observation
        }
    else:
        observations = np.array(
            [t.observation for traj in trajectories for t in traj], dtype=np.float32
        )
        next_observations = np.array(
            [t.next_observation for traj in trajectories for t in traj],
            dtype=np.float32,
        )
    actions = np.array(
        [t.action for traj in trajectories for t in traj], dtype=np.float32
    )
    rewards = np.array(
        [t.reward for traj in trajectories for t in traj], dtype=np.float32
    )
    discount = np.array(
        [t.discount for traj in trajectories for t in traj], dtype=np.float32
    )
    policy_extras = {
        key: np.array(
            [t.extras["policy_extras"][key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].extras["policy_extras"]
    }
    state_extras = {
        key: np.array(
            [t.extras["state_extras"][key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].extras["state_extras"]
        if key != "time"
    }
    infos = {
        "policy_extras": policy_extras,
        "state_extras": state_extras,
    }
    out = (
        observations,
        actions,
        rewards,
        discount,
        next_observations,
        infos,
    )
    return out
