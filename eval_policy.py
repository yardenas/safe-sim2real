import logging
from pathlib import Path

import hydra
import jax
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import mbpo, ppo, sac
from ss2r.common.wandb import get_wandb_checkpoint
from ss2r.common.logging import TrainingLogger

from train_brax import _validate_madrona_args

_LOG = logging.getLogger(__name__)

def average_metrics(metrics: list) -> dict:

    n_runs = len(metrics)
    if n_runs == 0:
        raise ValueError("Empty metrics list passed to function.")
    
    averaged_metrics = {}
    for metric in metrics[0].keys():
        avg_value = 0
        for i in range(n_runs):
            avg_value += metrics[i][metric]
        avg_value /= n_runs
        averaged_metrics[metric] = avg_value

    return averaged_metrics


def get_policy_fn(cfg):
    if cfg.training.wandb_id:
        checkpoint_path = get_wandb_checkpoint(cfg.training.wandb_id, cfg.wandb.entity)
    else:
        raise ValueError("wandb_id must be specified for evaluation.")
    if cfg.agent.name == "sac":
        train_fn = sac.get_train_fn(cfg, checkpoint_path=None, restore_checkpoint_path=checkpoint_path)
    elif cfg.agent.name == "ppo":
        train_fn = ppo.get_train_fn(cfg, checkpoint_path=None, restore_checkpoint_path=checkpoint_path)
    elif cfg.agent.name == "mbpo":
        train_fn = mbpo.get_train_fn(cfg, checkpoint_path=None, restore_checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn

@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Evaluating policy with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    # Avoid training by setting the number of timesteps to 0
    cfg.training.num_timesteps = 0
    cfg.agent.min_replay_size = -1

    if cfg.wandb.name_notes:
        notes = cfg.wandb.name_notes + '_'
    else:
        notes = ''

    cfg.wandb.name = 'eval_' + notes + cfg.wandb.name

    logger = TrainingLogger(cfg)

    # --- wandb setup ---
    wandb_handle = None
    if "wandb" in getattr(cfg, "writers", []):
        import wandb
        wandb.init(
            project=getattr(cfg, "wandb", {}).get("project", "ss2r"),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=True,
            **getattr(cfg, "wandb", {}),
        )
        wandb_handle = wandb

    # Create multiple runs using seeds
    try:
        seeds = range(cfg.eval_params.num_seeds)
    except:
        raise ValueError("Could not retrieve number of different seeds from config parameter 'eval_params.num_seeds'. Please provide it.")

    metrics = []
    for i, seed in enumerate(seeds):
        cfg.training.seed = seed

        train_fn = get_policy_fn(cfg)
        train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(cfg)
        use_vision = "use_vision" in cfg.agent and cfg.agent.use_vision
        train_env, eval_env = benchmark_suites.make(cfg, train_env_wrap_fn, eval_env_wrap_fn)
        if use_vision:
            _validate_madrona_args(
                train_env,
                eval_env,
                cfg.training.num_envs,
                cfg.training.num_eval_envs,
                cfg.training.action_repeat,
                cfg.environment.task_params.vision_config.render_batch_size,
            )
        with jax.disable_jit(not cfg.jit):
            make_policy, params, metrics_for_run = train_fn(
                environment=train_env,
                eval_env=eval_env,
                progress_fn=lambda *args, **kwargs: None,
            )
        _LOG.info(f"------Metrics for evaluation run {i}:")
        _LOG.info(metrics_for_run)
        metrics += [metrics_for_run]

    averaged_metrics = average_metrics(metrics)

    _LOG.info(f"------------Metrics averaged over all {cfg.training.num_seeds} runs:")
    _LOG.info(metrics_for_run)
    
    averaged_metrics['num_seeds'] = cfg.eval_params.num_seeds
    averaged_metrics['seeds'] = list(seeds)
    # Log metrics to wandb if enabled
    if wandb_handle is not None and averaged_metrics:
        wandb_handle.log(averaged_metrics)
    _LOG.info("Done evaluating.")

if __name__ == "__main__":
    main()
