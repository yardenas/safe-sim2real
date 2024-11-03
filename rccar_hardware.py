import logging

import hydra

from ss2r.benchmark_suites.rccar import hardware

_LOG = logging.getLogger(__name__)


def make_env():
    return None, None


def collect_trajectory(env, controller, policy):
    with hardware.start(controller):
        pass


def fetch_policy():
    return None


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="rccar_hardware")
def main(cfg):
    traj_count = 0
    policy_fn = fetch_policy()
    with hardware.connect(car_id=cfg.car_id) as controller:
        controller, env = make_env()
        while traj_count < cfg.num_trajectories:
            answer = input("Press Y/y when ready to collect trajectory")
            if not (answer == "Y" or answer == "y"):
                _LOG.info("Skipping trajectory")
                continue
            collect_trajectory(env)
            traj_count += 1
