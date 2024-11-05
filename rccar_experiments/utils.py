import time
from typing import NamedTuple

import jax
import numpy as np
from brax.training.types import Metrics, Policy

from ss2r.benchmark_suites.rccar import hardware, rccar


class Trajectory(NamedTuple):
    observation: list[jax.Array]
    action: list[jax.Array]
    reward: list[jax.Array]
    cost: list[jax.Array]


def collect_trajectory(
    env: rccar.RCCar, controller, policy: Policy, rng: jax.Array
) -> tuple[Metrics, Trajectory]:
    t = time.time()
    trajectory = Trajectory([], [], [], [])
    elapsed = []
    with hardware.start(controller):
        state = env.reset(rng)
        while not state.done:
            trajectory.observation.append(state.obs)
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            trajectory.action.append(action)
            state = env.step(state, action)
            trajectory.reward.append(state.reward)
            trajectory.cost.append(state.info["cost"])
            elapsed.append(state.info["elapsed_time"])
        trajectory.observation.append(state.obs)
    epoch_eval_time = time.time() - t
    eval_metrics = state.info["eval_metrics"].episode_metrics
    metrics = {
        "eval/walltime": epoch_eval_time,
        **eval_metrics,
        "eval/average_time": np.mean(elapsed),
    }
    return metrics, trajectory


class DummyPolicy:
    def __init__(self, actions) -> None:
        self.id = 0
        self.actions = actions

    def __call__(self, *arg, **kwargs):
        next_action = self.actions[self.id]
        self.id += 1
        return next_action, {}
