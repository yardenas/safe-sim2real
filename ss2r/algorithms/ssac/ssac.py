import functools

import equinox as eqx
import jax
from brax import envs
from brax.training import acting
from brax.training.types import Transition
from brax.envs.wrappers.gym import GymWrapper
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

from ss2r.algorithms.ssac import safe_actor_critic as sac
from ss2r.algorithms.ssac.replay_buffer import ReplayBuffer
from ss2r.rl.metrics import MetricsMonitor
from ss2r.rl.types import Report
from ss2r.rl.utils import PRNGSequence


@eqx.filter_jit
def policy(actor, observation, key):
    act = lambda o, k: actor.act(o, k)
    return eqx.filter_vmap(act)(
        observation, jax.random.split(key, observation.shape[0])
    ), {}


class SafeSAC:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
    ):
        self.prng = PRNGSequence(config.training.seed)
        self.config = config
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(
            config.agent.replay_buffer.capacity,
            config.training.seed,
            config.agent.batch_size,
        )
        self.actor_critic = sac.ActorCritic(
            observation_space, action_space, config, next(self.prng)
        )
        self.metrics_monitor = MetricsMonitor()

    def update(self):
        if len(self.replay_buffer) > self.config.agent.prefill:
            for batch in self.replay_buffer.sample(
                self.config.agent.num_grad_steps_per_step
            ):
                losses = self.actor_critic.update(batch, next(self.prng))
                log(losses, self.metrics_monitor)
            self.actor_critic.polyak(self.config.agent.polyak_rate)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.store(transition)

    def report(self) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        return Report(metrics=metrics)


def log(log_items, monitor):
    for k, v in log_items.items():
        monitor[k] = v.item()


def train(config, environment, progress_fn, checkpoint_logdir):
    dummy_gym_env = GymWrapper(environment)
    agent = SafeSAC(dummy_gym_env.observation_space, dummy_gym_env.action_space, config)
    local_key = jax.random.PRNGKey(config.training.seed)
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    env_keys = jax.random.split(env_key, config.training.num_envs)
    env = envs.training.wrap(
        environment, config.training.episode_length, config.training.action_repeat
    )
    eval_env = environment
    eval_env = envs.training.wrap(
        eval_env,
        episode_length=config.training.episode_length,
        action_repeat=config.training.action_repeat,
    )
    env_state = env.reset(env_keys)
    _, actor_static = eqx.partition(agent.actor_critic.actor, eqx.is_inexact_array)

    def make_policy(actor):
        actor = eqx.combine(actor, actor_static)
        return functools.partial(policy, actor)

    evaluator = acting.Evaluator(
        eval_env,
        make_policy,
        num_eval_envs=config.training.num_eval_envs,
        episode_length=config.training.episode_length,
        action_repeat=config.training.action_repeat,
        key=eval_key,
    )
    metrics = evaluator.run_evaluation(
        eqx.filter(agent.actor_critic.actor, eqx.is_array),
        training_metrics={},
    )
    progress_fn(0, metrics)
    step = 0
    steps_per_iteration = config.training.num_envs * config.training.action_repeat
    steps_per_eval = config.training.num_timesteps // config.training.num_evals
    next_eval_step = steps_per_eval
    iteration = 0
    eval_count = 0
    while (
        step < config.training.num_timesteps or eval_count < config.training.num_evals
    ):
        local_key, key = jax.random.split(local_key)
        env_state, transitions = acting.actor_step(
            env,
            env_state,
            functools.partial(policy, agent.actor_critic.actor),
            key,
            extra_fields=("truncation",),
        )
        agent.observe(transitions)
        agent.update()
        step += steps_per_iteration
        iteration += 1
        if step >= next_eval_step and eval_count < config.training.num_evals:
            local_key, key = jax.random.split(local_key)
            report = agent.report()
            metrics = evaluator.run_evaluation(
                eqx.filter(agent.actor_critic.actor, eqx.is_array), report.metrics
            )
            eval_count += 1
            next_eval_step += steps_per_eval
            current_epoch = eval_count
            progress_fn(current_epoch, metrics)
