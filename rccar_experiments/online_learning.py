from typing import Mapping

import hydra
import jax.nn as jnn
import jax.numpy as jnp
import wandb
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint

import ss2r.algorithms.mbpo.networks as mbpo_networks
from rccar_experiments.experiment_driver import ExperimentDriver
from rccar_experiments.utils import make_env
from ss2r.algorithms.mbpo import safety_filters
from ss2r.algorithms.mbpo.train import get_dict_normalizer_params
from ss2r.benchmark_suites.rccar import hardware
from ss2r.common.wandb import get_wandb_checkpoint


def fetch_wandb_policy(cfg, env):
    api = wandb.Api(overrides={"entity": cfg.wandb.entity})
    run = api.run(f"ss2r/{cfg.wandb_id}")
    run_config = run.config
    restore_checkpoint_path = get_wandb_checkpoint(cfg.wandb_id, cfg.wandb.entity)
    params = checkpoint.load(restore_checkpoint_path)
    if run_config["agent"]["normalize_observations"]:
        normalize = running_statistics.normalize
    else:
        normalize = lambda x, y: x
    network = mbpo_networks.make_mbpo_networks(
        observation_size=7,
        action_size=2,
        policy_hidden_layer_sizes=run_config["agent"]["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=run_config["agent"]["value_hidden_layer_sizes"],
        activation=jnn.swish,
        preprocess_observations_fn=normalize,
        safe=cfg.safe,
    )
    if cfg.safety_filter == "sooper":
        normalizer_params = params[0]
        obs_size = env.observation_size
        if isinstance(obs_size, Mapping):
            obs_shape = {
                k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
            }
        else:
            obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
        normalizer_params = running_statistics.init_state(obs_shape)
        if not isinstance(params[0].mean, dict):
            normalizer_params = get_dict_normalizer_params(params, normalizer_params)
        backup_policy_params = params[1]
        budget_scaling_fn = (
            lambda x: x
            * cfg.episode_length
            * (1.0 - run_config["agent"]["safety_discounting"])
            / run_config["training"]["action_repeat"]
        )
        return safety_filters.make_sooper_filter_fn(
            network, backup_policy_params, normalizer_params, budget_scaling_fn
        )
    else:
        return safety_filters.make_inference_fn(network)


@hydra.main(
    version_base=None,
    config_path="../ss2r/configs",
    config_name="rccar_online_learning",
)
def main(cfg):
    with (
        hardware.connect(
            car_id=cfg.car_id,
            port_number=cfg.port_number,
            control_frequency=cfg.control_frequency,
        ) as controller,
    ):
        env = make_env(cfg, controller)
        policy_factory = fetch_wandb_policy(cfg, env)
        driver = ExperimentDriver(cfg, controller, policy_factory, env)
        driver.run()


if __name__ == "__main__":
    main()
