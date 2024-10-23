import functools
import logging
import os

import hydra
from omegaconf import OmegaConf

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.benchmark_suites import make
from ss2r.rl.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


def get_train_fn(cfg):
    if cfg.agent.name == "sac":
        import ss2r.algorithms.sac.train as sac

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "safe",
                "render_episodes",
                "safety_budget",
                "train_domain_randomization",
                "eval_domain_randomization",
                "privileged",
            ]
        }
        hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
        del agent_cfg["name"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks, hidden_layer_sizes=hidden_layer_sizes
        )
        train_fn = functools.partial(
            sac.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            checkpoint_logdir=get_state_path(),
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


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
    train_env, eval_env, domain_randomization_params = make(cfg)
    train_fn = get_train_fn(cfg)
    make_inference_fn, params, _ = train_fn(
        environment=train_env,
        eval_env=eval_env,
        wrap_env=False,
        progress_fn=functools.partial(report, logger),
        domain_parameters=domain_randomization_params,
    )
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
