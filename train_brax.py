import functools
import logging
import os
from pathlib import Path

import hydra
import jax
import wandb
from omegaconf import OmegaConf

import ss2r.algorithms.sac.networks as sac_networks
from ss2r import benchmark_suites
from ss2r.algorithms.penalizers import (
    CRPO,
    AugmentedLagrangian,
    AugmentedLagrangianParams,
    CRPOParams,
    Lagrangian,
    LagrangianParams,
)
from ss2r.algorithms.ppo.wrappers import Saute
from ss2r.algorithms.sac import robustness as rb
from ss2r.algorithms.sac.data import get_collection_fn
from ss2r.algorithms.sac.wrappers import ModelDisagreement, SPiDR
from ss2r.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd() + "/cpkt"
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


def get_penalizer(cfg):
    if cfg.agent.penalizer.name == "lagrangian":
        penalizer = AugmentedLagrangian(cfg.agent.penalizer.penalty_multiplier_factor)
        penalizer_state = AugmentedLagrangianParams(
            cfg.agent.penalizer.lagrange_multiplier,
            cfg.agent.penalizer.penalty_multiplier,
        )
    elif cfg.agent.penalizer.name == "crpo":
        penalizer = CRPO(cfg.agent.penalizer.eta)
        penalizer_state = CRPOParams(cfg.agent.penalizer.burnin)
    elif cfg.agent.penalizer.name == "ppo_lagrangian":
        penalizer = Lagrangian(cfg.agent.penalizer.multiplier_lr)
        init_lagrange_multiplier = cfg.agent.penalizer.initial_lagrange_multiplier
        penalizer_state = LagrangianParams(
            init_lagrange_multiplier,
            penalizer.optimizer.init(init_lagrange_multiplier),
        )
    elif cfg.agent.penalizer.name == "saute":
        return None, None
    else:
        raise ValueError(f"Unknown penalizer {cfg.agent.penalizer.name}")
    return penalizer, penalizer_state


def get_cost_robustness(cfg):
    if (
        "cost_robustness" not in cfg.agent
        or cfg.agent.cost_robustness is None
        or cfg.agent.cost_robustness.name == "neutral"
    ):
        return rb.SACCost()
    if cfg.agent.cost_robustness.name == "ramu":
        del cfg.agent.cost_robustness.name
        robustness = rb.RAMU(**cfg.agent.cost_robustness)
    elif cfg.agent.cost_robustness.name == "ucb_cost":
        robustness = rb.UCBCost(
            cfg.agent.cost_robustness.cost_penalty, cfg.agent.cost_robustness.alpha
        )
    else:
        raise ValueError("Unknown robustness")
    return robustness


def get_reward_robustness(cfg):
    if (
        "reward_robustness" not in cfg.agent
        or cfg.agent.reward_robustness is None
        or cfg.agent.reward_robustness.name == "neutral"
    ):
        return rb.SACBase()
    if cfg.agent.reward_robustness.name == "ramu":
        del cfg.agent.reward_robustness.name
        robustness = rb.RAMUReward(**cfg.agent.reward_robustness)
    elif cfg.agent.reward_robustness.name == "lcb_reward":
        robustness = rb.LCBReward(cfg.agent.reward_robustness.reward_penalty)
    else:
        raise ValueError("Unknown robustness")
    return robustness


def get_wrap_env_fn(cfg):
    if "propagation" not in cfg.agent:
        out = lambda env: env, lambda env: env
    elif cfg.agent.propagation.name == "ts1":

        def fn(env):
            key = jax.random.PRNGKey(cfg.training.seed)
            env = SPiDR(
                env,
                benchmark_suites.prepare_randomization_fn(
                    key,
                    cfg.agent.propagation.num_envs,
                    cfg.environment.train_params,
                    cfg.environment.task_name,
                ),
                cfg.agent.propagation.num_envs,
            )
            env = ModelDisagreement(env)
            return env

        out = fn, lambda env: env
    else:
        raise ValueError("Propagation method not provided.")
    if "penalizer" in cfg.agent and cfg.agent.penalizer.name == "saute":

        def saute_train(env):
            env = out[0](env)
            env = Saute(
                env,
                cfg.training.episode_length,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                cfg.agent.penalizer.penalty,
                cfg.agent.penalizer.terminate,
                cfg.agent.penalizer.lambda_,
            )
            return env

        def saute_eval(env):
            env = out[1](env)
            env = Saute(
                env,
                cfg.training.episode_length,
                cfg.agent.safety_discounting,
                cfg.training.safety_budget,
                0.0,
                False,
                0.0,
            )
            return env

        out = saute_train, saute_eval
    return out


def get_train_fn(cfg):
    if cfg.training.wandb_id:
        restore_checkpoint_dir = get_wandb_checkpoint(cfg.training.wandb_id)
    else:
        restore_checkpoint_dir = None
    if cfg.agent.name == "sac":
        import jax.nn as jnn

        import ss2r.algorithms.sac.train as sac

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "render_episodes",
                "train_domain_randomization",
                "eval_domain_randomization",
                "render",
                "store_checkpoint",
                "value_privileged",
                "policy_privileged",
                "wandb_id",
            ]
        }
        policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
        value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        if "cost_robustness" in agent_cfg:
            del agent_cfg["cost_robustness"]
        if "reward_robustness" in agent_cfg:
            del agent_cfg["reward_robustness"]
        if "penalizer" in agent_cfg:
            del agent_cfg["penalizer"]
        if "propagation" in agent_cfg:
            del agent_cfg["propagation"]
        if "data_collection" in agent_cfg:
            del agent_cfg["data_collection"]
        value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
        policy_obs_key = (
            "privileged_state" if cfg.training.policy_privileged else "state"
        )
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            value_obs_key=value_obs_key,
            policy_obs_key=policy_obs_key,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        cost_robustness = get_cost_robustness(cfg)
        reward_robustness = get_reward_robustness(cfg)
        # TODO (yarden): refactor all of these things to be algo dependent.
        # Create algos in their init
        data_collection = get_collection_fn(cfg)
        train_fn = functools.partial(
            sac.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            checkpoint_logdir=get_state_path(),
            cost_robustness=cost_robustness,
            reward_robustness=reward_robustness,
            penalizer=penalizer,
            penalizer_params=penalizer_params,
            get_experience_fn=data_collection,
            restore_checkpoint_path=restore_checkpoint_dir,
        )
    elif cfg.agent.name == "ppo":
        import jax.nn as jnn

        from ss2r.algorithms.ppo import networks as ppo_networks
        from ss2r.algorithms.ppo import train as ppo

        agent_cfg = dict(cfg.agent)
        training_cfg = {
            k: v
            for k, v in cfg.training.items()
            if k
            not in [
                "render_episodes",
                "train_domain_randomization",
                "eval_domain_randomization",
                "render",
                "store_checkpoint",
            ]
        }
        policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
        value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
        policy_obs_key = (
            "privileged_state" if cfg.training.policy_privileged else "state"
        )
        cost_robustness = get_cost_robustness(cfg)
        del training_cfg["value_privileged"]
        del training_cfg["policy_privileged"]
        del agent_cfg["name"]
        if "cost_robustness" in agent_cfg:
            del agent_cfg["cost_robustness"]
        if "reward_robustness" in agent_cfg:
            del agent_cfg["reward_robustness"]
        if "penalizer" in agent_cfg:
            del agent_cfg["penalizer"]
        if "propagation" in agent_cfg:
            del agent_cfg["propagation"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            value_obs_key=value_obs_key,
            policy_obs_key=policy_obs_key,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        train_fn = functools.partial(
            ppo.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            restore_checkpoint_path=restore_checkpoint_dir,
            penalizer=penalizer,
            penalizer_params=penalizer_params,
        )
        # FIXME (yarden): that's a hack for now. Need to think of a
        # better way to implement this.
        if "penalizer" in cfg.agent and cfg.agent.penalizer.name == "saute":
            train_fn = functools.partial(
                train_fn,
                use_saute=cfg.training.safe,
            )
        if "propagation" in cfg.agent and cfg.agent.propagation.name == "ts1":
            train_fn = functools.partial(
                train_fn,
                use_disagreement=True,
                disagreement_scale=cost_robustness.lambda_,
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
    train_env_wrap_fn, eval_env_wrap_fn = get_wrap_env_fn(cfg)
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
