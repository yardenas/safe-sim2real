import functools
import logging
import os

import hydra
import jax
from brax.io import image
from brax.training.types import Policy, State
from omegaconf import OmegaConf

import ss2r.algorithms.sac.networks as sac_networks
from ss2r import benchmark_suites
from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.logging import TrainingLogger

_LOG = logging.getLogger(__name__)


def rollout(
    self,
    policy: Policy,
    steps: int,
    rng: int,
    state: State | None = None,
) -> tuple[State, State]:
    if state is None:
        rng, rng = jax.random.split(rng)
        keys = jax.random.split(rng, self.parallel_envs)
        state = self.environment.reset(keys)

    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = self.step(
            state,
            policy,
            current_key,
            extra_fields=("truncation",),
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(f, (state, rng), (), length=steps)
    return final_state, data


def render_video(sys, policy, steps, rng):
    _, trajectory = rollout(policy, steps, rng)
    trajectory = jax.tree_map(lambda x: x[:, 0], trajectory.extras["pipeline_state"])  # type: ignore
    trajectory = pytrees_unstack(trajectory)
    video = image.render_array(sys, trajectory)
    return video


def get_state_path() -> str:
    log_path = os.getcwd()
    return log_path


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
                "safe",
                "render_episodes",
                "safety_budget",
                "train_domain_randomization",
                "eval_domain_randomization",
                "privileged",
            ]
        }
        hidden_layer_sizes = agent_cfg.pop("hidden_layer_sizes")
        activation = getattr(jnn, agent_cfg.pop("activation"))
        del agent_cfg["name"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        train_fn = functools.partial(
            sac.train,
            **agent_cfg,
            **training_cfg,
            network_factory=network_factory,
            checkpoint_logdir=get_state_path(),
        )
    elif cfg.agent.name == "sac_lenart":
        import jax
        from jax.nn import swish
        from mbpo.optimizers.policy_optimizers.sac.sac_brax_env import SAC

        def train(environment, eval_env, wrap_env, progress_fn, domain_parameters):
            num_env_steps_between_updates = 1
            num_envs = 128
            optimizer = SAC(
                environment=environment,
                num_timesteps=500000,
                episode_length=100,
                action_repeat=1,
                num_env_steps_between_updates=num_env_steps_between_updates,
                num_envs=num_envs,
                num_eval_envs=128,
                lr_alpha=3e-4,
                lr_policy=3e-4,
                lr_q=3e-4,
                wd_alpha=0.0,
                wd_policy=0.0,
                wd_q=0.0,
                max_grad_norm=1e5,
                discounting=0.99,
                batch_size=128,
                num_evals=20,
                normalize_observations=True,
                reward_scaling=1.0,
                tau=0.005,
                min_replay_size=10**2,
                max_replay_size=10**5,
                grad_updates_per_step=num_env_steps_between_updates * num_envs,
                deterministic_eval=True,
                init_log_alpha=0.0,
                policy_hidden_layer_sizes=(64, 64),
                policy_activation=swish,
                critic_hidden_layer_sizes=(64, 64),
                critic_activation=swish,
                return_best_model=True,
                eval_environment=eval_env,
            )
            optimizer.run_training(
                jax.random.PRNGKey(cfg.training.seed), progress_fn=progress_fn
            )

        return train
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


def report(logger, num_steps, metrics):
    metrics = {
        "train/objective": float(metrics["eval/episode_reward"]),
        "train/sps": float(metrics["eval/sps"]),
    }
    logger.log(metrics, num_steps)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="config")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_env, eval_env, domain_randomization_params = benchmark_suites.make(cfg)
    train_fn = get_train_fn(cfg)
    make_policy, params, metrics = train_fn(
        environment=train_env,
        eval_env=eval_env,
        wrap_env=False,
        progress_fn=functools.partial(report, logger),
        domain_parameters=domain_randomization_params,
    )
    video = render_video(
        train_env.sys,
        make_policy(params, deterministic=True),
        cfg.training.episode_length,
        jax.random.PRNGKey(cfg.training.seed),
    )
    logger.log_video(video, 0)
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
