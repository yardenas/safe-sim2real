import jax
from brax.envs.wrappers.training import EpisodeWrapper
from brax.training.types import Metrics, Policy, Transition

from ss2r.benchmark_suites.rccar import hardware, rccar
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.rl.evaluation import ConstraintEvalWrapper


def collect_trajectory(
    env: rccar.RCCar, policy: Policy, rng: jax.Array
) -> tuple[Metrics, Transition]:
    state = env.reset(rng)
    transitions: list[Transition] = []
    while not state.done:
        rng, key = jax.random.split(rng)
        action, _ = policy(state.obs, key)
        obs = state.obs
        state = env.step(state, action)
        next_obs = state.obs
        transition = Transition(
            obs,
            action,
            state.reward,
            1 - state.done,
            next_obs,
            {"cost": state.info["cost"]},
        )
        transitions.append(transition)
    metrics = state.info["eval_metrics"].episode_metrics
    metrics = {key: float(value) for key, value in metrics.items()}
    return metrics, transitions


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
    train_car_params = task_cfg.pop("train_params")
    task_cfg.pop("eval_params")
    if controller is not None:
        dynamics = hardware.HardwareDynamics(controller=controller)
    else:
        dynamics = None
    env = rccar.RCCar(train_car_params["nominal"], **task_cfg, hardware=dynamics)
    env = EpisodeWrapper(env, cfg.episode_length, cfg.action_repeat)
    env = ConstraintEvalWrapper(env)
    return env
