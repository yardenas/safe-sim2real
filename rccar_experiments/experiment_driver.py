import logging
from collections import defaultdict

import jax

from rccar_experiments.session import Session
from rccar_experiments.transitions_server import TransitionsServer
from rccar_experiments.utils import collect_trajectory
from ss2r.benchmark_suites.rccar import hardware

_LOG = logging.getLogger(__name__)


class ExperimentDriver:
    def __init__(self, cfg, hardware_handle, rollout_policy_fn, env):
        self.session = Session(filename=cfg.session_id, directory="experiment_sessions")
        num_steps = len(self.session.steps)
        if num_steps != 0:
            seed = num_steps
        else:
            seed = cfg.seed
        self.run_id = num_steps
        self.key = jax.random.PRNGKey(seed)
        self.episode_length = cfg.episode_length
        self.transitions_server = TransitionsServer(self, safe_mode=True)
        self.hardware_handle = hardware_handle
        self.env = env
        self.rollout_policy_fn = rollout_policy_fn
        _LOG.info("Experiment driver initialized.")

    def run(self):
        self.transitions_server.loop()

    def sample_trajectory(self, policy):
        _LOG.info(f"Starting trajectory sampling... Run id: {self.run_id}")
        self.key, key = jax.random.split(self.key)
        policy = jax.jit(policy)
        with hardware.start(self.hardware_handle):
            _, trajectory = collect_trajectory(
                self.env, policy, key, self.episode_length
            )
        self.summarize_trial(trajectory)
        return trajectory

    def summarize_trial(self, transitions):
        infos = [transition.extras["state_extras"] for transition in transitions]
        table_data = defaultdict(float)
        for info in infos:
            for key, value in info.items():
                table_data[key] += value
        table_data["steps"] = len(infos)
        table_data["reward"] = float(
            sum(transition.reward for transition in transitions)
        )
        table_data["cost"] = sum(info["cost"] for info in infos)
        table_data["terminated"] = (
            1 - transitions[-1].discount and not infos[-1]["truncation"]
        )
        _LOG.info(
            f"Total reward: {table_data['reward']}\nTotal cost: {table_data['cost']}\n{_format_reward_summary(table_data)}"
        )
        self.session.update(table_data)
        self.run_id += 1

    @property
    def robot_ok(self):
        return True


def _format_reward_summary(table_data):
    lines = []
    header = f"{'Reward Component':<20} {'Total Value':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, value in table_data.items():
        lines.append(f"{key:<20} {value:>12.2f}")
    return "\n".join(lines)
