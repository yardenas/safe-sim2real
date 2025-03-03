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
        or cfg.agent.cost_robustness is None
        or cfg.agent.cost_robustness.name == "neutral"
    ):
        return rb.SACCost()
    if cfg.agent.cost_robustness.name == "ramu":
        del cfg.agent.cost_robustness.name
        robustness = rb.RAMU(**cfg.agent.cost_robustness)
    elif cfg.agent.cost_robustness.name == "ucb_cost":
        assert cfg.agent.propagation == "ts1"
        robustness = rb.UCBCost(cfg.agent.cost_robustness.cost_penalty)
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
        assert cfg.agent.propagation == "ts1"
        robustness = rb.LCBReward(cfg.agent.reward_robustness.reward_penalty)
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
                "value_privileged",
                "policy_privileged",
            ]
        }
        hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        if "cost_robustness" in agent_cfg:
            del agent_cfg["cost_robustness"]
        if "reward_robustness" in agent_cfg:
            del agent_cfg["reward_robustness"]
        if "penalizer" in agent_cfg:
            del agent_cfg["penalizer"]
        value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
        policy_obs_key = (
            "privileged_state" if cfg.training.policy_privileged else "state"
        )
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            value_obs_key=value_obs_key,
            policy_obs_key=policy_obs_key,
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
    elif cfg.agent.name == "ppo":
        from mujoco_playground.config import locomotion_params

        ppo_params = locomotion_params.brax_ppo_config(cfg.environment.task_name)
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo

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
            wrap_env=False,
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
