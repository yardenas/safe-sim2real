# Script to evaluate a policy using a checkpoint from Weights & Biases (wandb).
# It can be run standalone or using the wrapper script eval_policy_seeds.py.
# Example usage: python eval_policy.py +experiment=cartpole_vision training.render=false training.safe=true wandb.entity=ENTITY training.wandb_id=ID +cfg.training.seed=0
import json
import os
import logging
from pathlib import Path

import hydra
import jax
import numpy as np
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

def to_serializable(val):
    # Recursively convert JAX/NumPy arrays to Python scalars/lists
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if hasattr(val, "tolist"):
        return val.tolist()
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    return val

@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Evaluating policy with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    # Avoid training by setting the number of timesteps to 0
    cfg.training.num_timesteps = 0
    cfg.agent.min_replay_size = -1

    # --- wandb setup ---
    wandb_handle = None
    if "wandb" in getattr(cfg, "writers", []):
        import wandb
        cfg.wandb.name = 'eval_policy_' + cfg.wandb.name
        wandb.init(
            project=getattr(cfg, "wandb", {}).get("project", "ss2r"),
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
        # Save metrics to a JSON file
        if metrics:
            serializable_metrics = to_serializable(metrics)
            path = os.path.abspath(os.path.dirname(__file__))
            metrics_path = os.path.join(path, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(serializable_metrics, f)
    _LOG.info("Done evaluating policy.")

if __name__ == "__main__":
    main()
