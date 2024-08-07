import logging
import functools

from brax import envs
from brax.io import model
import brax.training.agents.ppo.train as ppo
from brax.training.agents.ppo import networks as ppo_networks
import hydra
from omegaconf import OmegaConf

from ss2r.benchmark_suites.utils import get_task_config
from ss2r.benchmark_suites.brax import randomization_fns
from ss2r.rl.logging import TrainingLogger
from ss2r.rl.trainer import get_state_path


_LOG = logging.getLogger(__name__)


def get_environment(cfg):
    task_cfg = get_task_config(cfg)
    env = envs.get_environment(task_cfg.task_name, backend="generalized")
    if cfg.environment.brax.domain_randomization:
        randomize_fn = functools.partial(
            randomization_fns[task_cfg.task_name], cfg=task_cfg
        )
    else:
        randomize_fn = None
    return env, randomize_fn


def checkpoint(current_step, make_policy, params):
    model.save_params(get_state_path(), params)


def report(logger, num_steps, metrics):
    metrics = {
        "train/objective": metrics["eval/episode_reward"],
        "train/sps": metrics["training/sps"],
    }
    logger.log(metrics, num_steps)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="config")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    policy_net_size = cfg.agent.pop("policy_layer_sizes")
    value_net_size = cfg.agent.pop("value_layer_sizes")
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_net_size,
        value_hidden_layer_sizes=value_net_size,
    )
    make_inference_fn, params, _ = ppo.train(
        **cfg.agent,
        progress_fn=functools.partial(report, logger),
        restore_checkpoint_path=get_state_path(),
        policy_params_fn=checkpoint,
        network_factory=network_factory,
    )
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
