import time
from typing import NamedTuple

import jax
import numpy as np
from brax.envs.wrappers.training import EpisodeWrapper
from brax.training.types import Metrics, Policy

from ss2r.benchmark_suites.rccar import hardware, rccar
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.benchmark_suites.wrappers import (
    ActionObservationDelayWrapper,
    FrameActionStack,
)
from ss2r.rl.evaluation import ConstraintEvalWrapper


class Trajectory(NamedTuple):
    observation: list[jax.Array]
    action: list[jax.Array]
    reward: list[jax.Array]
    cost: list[jax.Array]


def collect_trajectory(
    env: rccar.RCCar, policy: Policy, rng: jax.Array
) -> tuple[Metrics, Trajectory]:
    t = time.time()
    trajectory = Trajectory([], [], [], [])
    elapsed = []
    delay = []
    state = env.reset(rng)
    while not state.done:
        trajectory.observation.append(state.obs)
        rng, key = jax.random.split(rng)
        tp = time.time()
        action, _ = policy(state.obs, key)
        delay.append(time.time() - tp)
        trajectory.action.append(action)
        state = env.step(state, action)
        trajectory.reward.append(state.reward)
        trajectory.cost.append(state.info["cost"])
        elapsed.append(state.info.get("elapsed_time", 0.0))
    trajectory.observation.append(state.obs)
    epoch_eval_time = time.time() - t
    eval_metrics = state.info["eval_metrics"].episode_metrics
    metrics = {
        "eval/walltime": epoch_eval_time,
        **eval_metrics,
        "eval/average_time": np.mean(elapsed),
        "eval/average_delay": np.mean(delay),
    }
    metrics = {key: float(value) for key, value in metrics.items()}
    return metrics, trajectory


class DummyPolicy:
    def __init__(self, actions) -> None:
        self.id = 0
        self.actions = actions

    def __call__(self, *arg, **kwargs):
        next_action = self.actions[self.id]
        self.id = (self.id + 1) % len(self.actions)
        return next_action, {}


def make_env(cfg, controller=None):
    task_cfg = dict(get_task_config(cfg))
    task_cfg.pop("domain_name")
    task_cfg.pop("task_name")
    train_car_params = task_cfg.pop("train_car_params")
    task_cfg.pop("eval_car_params")
    if controller is not None:
        dynamics = hardware.HardwareDynamics(controller=controller)
    else:
        dynamics = None
    action_delay, obs_delay = (
        task_cfg.pop("action_delay"),
        task_cfg.pop("observation_delay"),
    )
    sliding_window = task_cfg.pop("sliding_window")
    env = rccar.RCCar(train_car_params["nominal"], **task_cfg, hardware=dynamics)
    if action_delay > 0 or obs_delay > 0:
        env = ActionObservationDelayWrapper(
            env, action_delay=action_delay, obs_delay=obs_delay
        )
    if sliding_window > 0:
        env = FrameActionStack(env, num_stack=sliding_window)
    env = EpisodeWrapper(env, cfg.episode_length, cfg.action_repeat)
    env = ConstraintEvalWrapper(env)
    return env
