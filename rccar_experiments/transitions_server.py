import logging
import pickle
import time

import numpy as np
import zmq

_LOG = logging.getLogger(__name__)


class TransitionsServer:
    def __init__(self, experiment_driver, safe_mode=False, address="tcp://*:5559"):
        self.experiment_driver = experiment_driver
        self.address = address
        self.safe_mode = safe_mode

    def loop(self):
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REP) as socket:
                socket.bind(self.address)
                while True:
                    message = socket.recv()
                    policy, num_steps = pickle.loads(message)
                    if num_steps < self.experiment_driver.trajectory_length:
                        _LOG.error("Invalid num_steps: {}".format(num_steps))
                    trials = self.run(policy, num_steps)
                    if trials is None:
                        continue
                    socket.send(pickle.dumps(trials))

    def run(self, policy, num_steps):
        trials = []
        num_transitions = 0
        while num_transitions < num_steps:
            trial = self.do_trial(policy)
            new_num_transitions = len(trial)
            if num_transitions + new_num_transitions > num_steps:
                trial = trial[: num_steps - num_transitions]
                trial[-1].info["truncation"] = True
                _LOG.info("Truncating trajectory")
            num_transitions += len(trial)
            trials.append(trial)
            _LOG.info("Completed trial")
        transitions = flatten_trajectories(trials)
        assert (
            len(transitions[2]) == num_steps
        ), f"Expected {num_steps} transitions, got {len(transitions)}"
        return transitions

    def do_trial(self, policy):
        _LOG.info("Starting sampling")
        if self.safe_mode:
            while not self.experiment_driver.running:
                _LOG.info("Waiting for command to start sampling...")
                time.sleep(2.5)
        else:
            time.sleep(2.5)
            while not self.experiment_driver.robot_ok:
                _LOG.info("Waiting the robot to be ready...")
                time.sleep(2.5)
            self.load_policy(policy)
            self.experiment_driver.start_sampling_callback(req, res)
        while self.experiment_driver.running:
            time.sleep(0.1)
        _LOG.info("Sampling finished")
        trajectory = self.experiment_driver.get_trajectory()
        return trajectory

    def load_policy(self, policy):
        with open(_ONNX_SAVE_PATH, "wb") as f:
            f.write(policy)
        while True:
            while not self.experiment_driver.fsm_state == 2:
                _LOG.info("Waiting for robot to be in walking state...")
                time.sleep(2.5)
            success = self.experiment_driver.update_policy(_ONNX_SAVE_PATH)
            if not success:
                _LOG.error("Failed to update policy")
                time.sleep(2.5)
            else:
                return


def flatten_trajectories(trajectories):
    observations = {
        key: np.array(
            [t.observation[key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].observation
    }
    actions = np.array(
        [t.action for traj in trajectories for t in traj], dtype=np.float32
    )
    rewards = np.array(
        [t.reward for traj in trajectories for t in traj], dtype=np.float32
    )
    next_observations = {
        key: np.array(
            [t.next_observation[key] for traj in trajectories for t in traj],
            dtype=np.float32,
        )
        for key in trajectories[0][0].next_observation
    }
    dones = np.array([t.done for traj in trajectories for t in traj], dtype=np.float32)
    infos = {
        key: np.array(
            [t.info[key] for traj in trajectories for t in traj], dtype=np.float32
        )
        for key in trajectories[0][0].info
    }
    return (
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        infos,
    )
