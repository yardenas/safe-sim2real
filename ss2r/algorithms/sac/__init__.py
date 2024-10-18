import equinox as eqx
import jax
import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

from .agent import safe_actor_critic as sac
from .agent.replay_buffer import ReplayBuffer
from .rl.epoch_summary import EpochSummary
from .rl.metrics import MetricsMonitor
from .rl.trajectory import Transition
from .rl.types import FloatArray, Report
from .rl.utils import PRNGSequence


@eqx.filter_jit
def policy(actor, observation, key):
    act = lambda o, k: actor.act(o, k)
    return eqx.filter_vmap(act)(
        observation, jax.random.split(key, observation.shape[0])
    )


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

    def __call__(self, observation: FloatArray, train: bool = True) -> FloatArray:
        if len(self.replay_buffer) > self.config.agent.prefill:
            for batch in self.replay_buffer.sample(self.config.training.parallel_envs):
                losses = self.actor_critic.update(batch, next(self.prng))
                log(losses, self.metrics_monitor)
            self.actor_critic.polyak(self.config.agent.polyak_rate)
        action = policy(self.actor_critic.actor, observation, next(self.prng))
        return np.asarray(action)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.store(transition)

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        return Report(metrics=metrics)


def log(log_items, monitor):
    for k, v in log_items.items():
        monitor[k] = v.item()
