import logging
import os
import pickle

import hydra
import jax
import jax.nn as jnn
import numpy as np
import wandb
from brax.io import model
from brax.training.acme import running_statistics

from rccar_experiments.utils import DummyPolicy, collect_trajectory, make_env
from ss2r.algorithms.sac.networks import make_inference_fn, make_sac_networks
from ss2r.benchmark_suites.rccar import hardware
from ss2r.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def fetch_wandb_policy(run_id):
    api = wandb.Api()
    run = api.run(f"ss2r/{run_id}")
    policy_artifact = api.artifact(f"ss2r/policy:{run_id}")
    policy_dir = policy_artifact.download()
    path = os.path.join(policy_dir, "policy.pkl")
    policy_params = model.load_params(path)
    config = run.config
    activation = getattr(jnn, config["agent"]["activation"])
    if config["agent"]["normalize_observations"]:
        normalize = running_statistics.normalize
    else:
        normalize = lambda x, y: x
    sac_network = make_sac_networks(
        observation_size=7,
        action_size=2,
        preprocess_observations_fn=normalize,
        policy_hidden_layer_sizes=config["agent"]["policy_hidden_layer_sizes"],
        activation=activation,
    )
    make_policy = make_inference_fn(sac_network)
    return make_policy(policy_params, True)


def load_recorded_policy(path):
    with open(path, "rb") as f:
        rec_traj = pickle.load(f)
        actions = rec_traj.action
    return DummyPolicy(actions)


def save_trajectory(trajectory, path):
    trajectory = jax.tree.map(np.asarray, trajectory)
    with open(path, "wb") as f:
        pickle.dump(trajectory, f)


@hydra.main(
    version_base=None, config_path="../ss2r/configs", config_name="rccar_hardware"
)
def main(cfg):
    traj_count = 0
    rng = jax.random.PRNGKey(cfg.seed)
    logger = TrainingLogger(cfg)
    with (
        hardware.connect(
            car_id=cfg.car_id,
            port_number=cfg.port_number,
            control_frequency=cfg.control_frequency,
        ) as controller,
        jax.disable_jit(),
    ):
        if cfg.policy_id is not None:
            assert cfg.playback_policy is None
            policy_fn = fetch_wandb_policy(cfg.policy_id)
        else:
            policy_fn = load_recorded_policy(cfg.playback_policy)
            cfg.episode_length = len(policy_fn.actions)
        if cfg.no_controller:
            env = make_env(cfg)
        else:
            env = make_env(cfg, controller)
        while traj_count < cfg.num_trajectories:
            answer = input("Press Y/y when ready to collect trajectory. N/n to stop.\n")
            if answer == "N" or answer == "n":
                _LOG.info("Stopping data collection.")
                break
            if not (answer == "Y" or answer == "y"):
                _LOG.info("Skipping trajectory")
                continue
            _LOG.info(f"Collecting trajectory {traj_count}")
            rng, key = jax.random.split(rng)
            with hardware.start(controller):
                metrics, trajectory = collect_trajectory(env, policy_fn, key)
            logger.log(metrics, traj_count)
            _LOG.info(f"Done trajectory: {traj_count}")
            if cfg.out_path_name is not None:
                save_trajectory(trajectory, f"{cfg.out_path_name}-{traj_count}.pkl")
            traj_count += 1


if __name__ == "__main__":
    main()
