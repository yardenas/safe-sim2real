import functools
import logging
import os
from pathlib import Path

import hydra
import jax
import wandb
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import mbpo, ppo, sac
from ss2r.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd() + "/ckpt"
    return log_path


def locate_last_checkpoint() -> Path | None:
    ckpt_dir = Path(get_state_path())
    # Get all directories or files that match the 12-digit pattern
    checkpoints = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 12
    ]
    if not checkpoints:
        return None  # No checkpoints found
    # Sort by step number (converted from the directory name)
    latest_ckpt = max(checkpoints, key=lambda p: int(p.name))
    return latest_ckpt


def get_wandb_checkpoint(run_id):
    api = wandb.Api()
    artifact = api.artifact(f"ss2r/checkpoint:{run_id}")
    download_dir = artifact.download(f"{get_state_path()}/{run_id}")
    return download_dir


def get_train_fn(cfg):
    if cfg.training.wandb_id:
        restore_checkpoint_path = get_wandb_checkpoint(cfg.training.wandb_id)
    else:
        restore_checkpoint_path = None
    if cfg.agent.name == "sac":
        train_fn = sac.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "ppo":
        train_fn = ppo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "mbpo":
        train_fn = mbpo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
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
    train_fn = get_train_fn(cfg)
    train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(cfg)
    train_env, eval_env = benchmark_suites.make(
        cfg, train_env_wrap_fn, eval_env_wrap_fn
    )
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=train_env,
            eval_env=eval_env,
            progress_fn=functools.partial(report, logger, steps),
        )
        if cfg.training.render:
            rng = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), cfg.training.num_eval_envs
            )
            if len(params) != 2:
                policy_params = params[:2]
            else:
                policy_params = params
            if cfg.agent.name == "mbpo" and cfg.agent.safe:
                policy_params = (
                    policy_params[0],
                    policy_params[1],
                    policy_params[2],
                    policy_params[4],
                )
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(policy_params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            logger.log_video(video, steps.count, "eval/video")
        if cfg.training.store_checkpoint:
            artifacts = locate_last_checkpoint()
            if artifacts:
                logger.log_artifact(artifacts, "model", "checkpoint")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
