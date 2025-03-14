import functools
import logging
import os

import hydra
import jax
from brax.io import model
from mujoco_playground import registry, wrapper
from omegaconf import OmegaConf

from ss2r.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


env_name = "Go1JoystickFlatTerrain"
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
eval_env = registry.load(env_name, config=env_cfg)


def get_train_fn():
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground.config import locomotion_params

    ppo_params = locomotion_params.brax_ppo_config(env_name)
    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
    train_fn = functools.partial(
        ppo.train,
        **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer,
    )
    return train_fn


class Counter:
    def __init__(self):
        self.count = 0


def report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_fn = get_train_fn()
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=env,
            eval_env=eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
            progress_fn=functools.partial(report, logger, steps),
        )
        if cfg.training.store_policy:
            path = get_state_path() + "/policy.pkl"
            model.save_params(get_state_path() + "/policy.pkl", params)
            logger.log_artifact(path, "model", "policy")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
