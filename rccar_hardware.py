import logging
import os
import time

import cloudpickle
import hydra
import jax
import omegaconf
import wandb
from brax.envs.wrappers.training import EpisodeWrapper
from brax.training.types import Metrics, Policy

from ss2r.benchmark_suites.rccar import hardware, rccar
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.benchmark_suites.wrappers import ActionObservationDelayWrapper
from ss2r.rl.evaluation import ConstraintEvalWrapper

_LOG = logging.getLogger(__name__)


def make_env(controller, cfg):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_car_params")
    dynamics = hardware.HardwareDynamics(
        controller=controller, max_throttle=task_cfg.max_throttle
    )
    env = rccar.RCCar(train_car_params["nominal"], **task_cfg, hardware=dynamics)
    env = EpisodeWrapper(env, cfg.episode_length, cfg.action_repeat)
    if task_cfg.action_delay > 0 or task_cfg.observation_delay > 0:
        env = ActionObservationDelayWrapper(
            env,
            action_delay=task_cfg.action_delay,
            obs_delay=task_cfg.observation_delay,
        )
    env = ConstraintEvalWrapper(env)
    return env


def collect_trajectory(
    env: rccar.RCCar, controller, policy: Policy, rng: jax.Array
) -> Metrics:
    t = time.time()
    with hardware.start(controller):
        state = env.reset(rng)
        while not state.done:
            rng, key = jax.random.split(rng)
            action = policy(state.obs, key)
            state = env.step(state, action)
    epoch_eval_time = time.time() - t
    eval_metrics = state.info["eval_metrics"]
    metrics = {"eval/walltime": epoch_eval_time, **eval_metrics}
    return metrics


def fetch_policy(policy_id, run):
    policy_artifact = run.use_artifact(f"policy:{policy_id}")
    policy_dir = policy_artifact.download()
    path = os.path.join(policy_dir, "policy.pkl")
    with open(path, "rb") as f:
        data = cloudpickle.load(f)
    make_policy = data["make_policy"]
    policy_params = data["params"]
    return make_policy(policy_params, True)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="rccar_hardware")
def main(cfg):
    traj_count = 0
    rng = jax.random.PRNGKey(cfg.seed)
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    with wandb.init(
        project="ss2r", resume=True, config=config_dict, **cfg.wandb
    ) as run, hardware.connect(
        car_id=cfg.car_id,
        port_number=cfg.port_number,
        control_frequency=cfg.control_frequency,
    ) as controller:
        policy_fn = fetch_policy(cfg.policy_id, run)
        env = make_env(controller, cfg)
        while traj_count < cfg.num_trajectories:
            answer = input("Press Y/y when ready to collect trajectory")
            if not (answer == "Y" or answer == "y"):
                _LOG.info("Skipping trajectory")
                continue
            _LOG.info(f"Collecting trajectory {traj_count}")
            rng, key = jax.random.split(rng)
            metrics = collect_trajectory(env, controller, policy_fn, key)
            print(metrics)
            _LOG.info(f"Done trajectory: {traj_count}")
            traj_count += 1


if __name__ == "__main__":
    main()
