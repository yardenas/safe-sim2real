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
    logger = TrainingLogger(cfg)

    # --- wandb setup ---
    wandb_handle = None
    if "wandb" in getattr(cfg, "writers", []):
        import wandb
        wandb.init(
            project=getattr(cfg, "wandb", {}).get("project", "ss2r"),
            # entity=getattr(cfg, "wandb", {}).get("entity", None),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=True,
            **getattr(cfg, "wandb", {}),
        )
        wandb_handle = wandb

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
        make_policy, params, metrics = train_fn(
            environment=train_env,
            eval_env=eval_env,
            progress_fn=lambda *args, **kwargs: None,
        )
        _LOG.info("------------Metrics:")
        _LOG.info(metrics)
        # Log metrics to wandb if enabled
        if wandb_handle is not None and metrics:
            wandb_handle.log(metrics)
    _LOG.info("Done evaluating.")

if __name__ == "__main__":
    main()
