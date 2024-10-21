import functools
import logging

import hydra
from brax import envs
from omegaconf import OmegaConf

import ss2r.algorithms.sac.train as sac
from ss2r.benchmark_suites.brax import randomization_fns
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.rl.logging import TrainingLogger
from ss2r.rl.trainer import get_state_path
import ss2r.algorithms.sac.networks as sac_networks

_LOG = logging.getLogger(__name__)


def get_environment(cfg):
    task_cfg = get_task_config(cfg)
    env = envs.get_environment(task_cfg.task_name, backend=cfg.environment.brax.backend)
    if cfg.environment.brax.domain_randomization:
        randomize_fn = lambda sys, rng: randomization_fns[task_cfg.task_name](
            sys, rng, task_cfg
        )
    else:
        randomize_fn = None

    return env, randomize_fn


def report(logger, num_steps, metrics):
    metrics = {
        "train/objective": float(metrics["eval/episode_reward"]),
        "train/sps": float(metrics["eval/sps"]),
    }
    logger.log(metrics, num_steps)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="config")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    agent_cfg = dict(cfg.agent)
    hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
    network_factory = functools.partial(
        sac_networks.make_sac_networks, hidden_layer_sizes=hidden_layer_sizes
    )
    environment, randomization_fn = get_environment(cfg)
    make_inference_fn, params, _ = sac.train(
        **agent_cfg,
        environment=environment,
        progress_fn=functools.partial(report, logger),
        checkpoint_logdir=get_state_path(),
        network_factory=network_factory,
        randomization_fn=randomization_fn,
    )
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
