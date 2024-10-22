import functools
import logging
import os

import hydra
import jax
from brax import envs
from omegaconf import OmegaConf

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.benchmark_suites.brax import randomization_fns
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.rl.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


def get_environment(cfg):
    task_cfg = get_task_config(cfg)
    env = envs.get_environment(task_cfg.task_name, backend=cfg.environment.brax.backend)
    train_key, eval_key = jax.random.split(jax.random.PRNGKey(cfg.training.seed))

    def prepare_randomization_fn(key, num_envs):
        randomize_fn = lambda sys, rng: randomization_fns[task_cfg.task_name](
            sys, rng, task_cfg
        )
        v_randomization_fn = functools.partial(
            randomize_fn, rng=jax.random.split(key, num_envs)
        )
        vf_randomization_fn = lambda sys: v_randomization_fn(sys)[:-1]  # type: ignore
        return vf_randomization_fn

    train_randomization_fn = (
        prepare_randomization_fn(train_key, cfg.training.num_envs)
        if cfg.training.train_domain_randomization
        else None
    )
    train_env = envs.training.wrap(
        env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=train_randomization_fn,
    )
    eval_env = envs.training.wrap(
        env,
        episode_length=cfg.training.episode_length,
        action_repeat=cfg.training.action_repeat,
        randomization_fn=prepare_randomization_fn(eval_key, cfg.training.num_eval_envs)
        if cfg.training.eval_domain_randomization
        else None,
    )
    if cfg.training.train_domain_randomization and cfg.training.privileged:
        domain_parameters = train_randomization_fn(train_env.sys)
    else:
        domain_parameters = None
    return train_env, eval_env, domain_parameters


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
                "eval_domain_randomization",
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
    train_env, eval_env, domain_randomization_params = get_environment(cfg)
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
