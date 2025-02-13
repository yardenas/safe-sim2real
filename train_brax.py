import functools
import logging
import os

import hydra
import jax
from brax.io import model
from omegaconf import OmegaConf

import ss2r.algorithms.sac.networks as sac_networks
from ss2r import benchmark_suites
from ss2r.algorithms.sac import robustness as rb
from ss2r.algorithms.sac.penalizers import CRPO, AugmentedLagrangian, LagrangianParams
from ss2r.common.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


def get_penalizer(cfg):
    if cfg.agent.penalizer.name == "lagrangian":
        penalizer = AugmentedLagrangian(cfg.agent.penalizer.penalty_multiplier_factor)
        penalizer_state = LagrangianParams(
            cfg.agent.penalizer.lagrange_multiplier,
            cfg.agent.penalizer.penalty_multiplier,
        )
    elif cfg.agent.penalizer.name == "crpo":
        penalizer = CRPO(cfg.agent.penalizer.eta)
        penalizer_state = None
    else:
        raise ValueError(f"Unknown penalizer {cfg.agent.penalizer.name}")
    return penalizer, penalizer_state


def get_cost_robustness(cfg):
    if (
        "cost_robustness" not in cfg.agent
        or cfg.agent.robustness is None
        or cfg.agent.robustness.name == "neutral"
    ):
        return rb.SACCost()
    assert cfg.agent.propagation == "ts1"
    if cfg.agent.robustness.name == "ramu":
        robustness = rb.RAMU(**cfg.agent.robustness)
    elif cfg.agent.robustness.name == "ucb_cost":
        robustness = rb.UCBCost(cfg.agent.robustness.cost_penalty)
    else:
        raise ValueError("Unknown robustness")
    return robustness


def get_reward_robustness(cfg):
    if (
        "reward_robustness" not in cfg.agent
        or cfg.agent.robustness is None
        or cfg.agent.robustness.name == "neutral"
    ):
        return rb.SACReward()
    assert cfg.agent.propagation == "ts1"
    if cfg.agent.robustness.name == "ramu":
        robustness = rb.RAMUReward(**cfg.agent.robustness)
    elif cfg.agent.robustness.name == "lcb_reward":
        robustness = rb.LCBReward(cfg.agent.robustness.reward_penalty)
    else:
        raise ValueError("Unknown robustness")
    return robustness


def get_train_fn(cfg):
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
                "store_policy",
            ]
        }
        hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        if "robustness" in agent_cfg:
            del agent_cfg["robustness"]
        if "penalizer" in agent_cfg:
            del agent_cfg["penalizer"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        penalizer, penalizer_params = get_penalizer(cfg)
        cost_robustness = get_cost_robustness(cfg)
        reward_robustness = get_reward_robustness(cfg)
        train_fn = functools.partial(
            sac.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            checkpoint_logdir=f"{get_state_path()}/ckpt",
            cost_robustness=cost_robustness,
            reward_robustness=reward_robustness,
            penalizer=penalizer,
            penalizer_params=penalizer_params,
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
    train_env, eval_env = benchmark_suites.make(cfg)
    train_fn = get_train_fn(cfg)
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
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            logger.log_video(video, steps.count, "eval/video")
        if cfg.training.store_policy:
            path = get_state_path() + "/policy.pkl"
            model.save_params(get_state_path() + "/policy.pkl", params)
            logger.log_artifact(path, "model", "policy")
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
