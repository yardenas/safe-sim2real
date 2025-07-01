import hydra
import jax
import jax.nn as jnn
import wandb
from brax.training.acme import running_statistics
from brax.training.agents.sac import checkpoint

import ss2r.algorithms.mbpo.networks as mbpo_networks
from rccar_experiments.experiment_driver import ExperimentDriver
from ss2r.algorithms.mbpo import safety_filters
from ss2r.benchmark_suites.rccar import hardware
from ss2r.common.wandb import get_wandb_checkpoint


def fetch_wandb_policy(cfg):
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
        activation=jnn.swish,
        value_obs_key=jax.random.PRNGKey(0),
        policy_obs_key=jax.random.PRNGKey(0),
        preprocess_observations_fn=normalize,
        safe=cfg.safe,
    )
    if cfg.safety_filter == "sooper":
        normalizer_params = params[0]
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
        jax.disable_jit(),
    ):
        policy_factory = fetch_wandb_policy(cfg)
        driver = ExperimentDriver(cfg, controller, policy_factory)
        driver.run()


if __name__ == "__main__":
    main()
