import logging

import hydra

from ss2r.benchmark_suites.rccar import hardware

_LOG = logging.getLogger(__name__)


def make_env():
    return None, None


def collect_trajectory():
    pass


def fetch_policy():
    return None


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="rccar_hardware")
def main(cfg):
    traj_count = 0
    controller, env = make_env()
    policy_fn = fetch_policy()
    while traj_count < cfg.num_trajectories:
        answer = input("Press Y/y when ready to collect trajectory")
        if not (answer == "Y" or answer == "y"):
            _LOG.info("Skipping trajectory")
            continue
        with hardware.start(controller):
            collect_trajectory(env)
        traj_count += 1
