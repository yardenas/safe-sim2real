from omegaconf import DictConfig

from ss2r.benchmark_suites.dm_control import ENVIRONMENTS as dm_control_envs
from ss2r.benchmark_suites.utils import get_domain_and_task
from ss2r.rl.types import EnvironmentFactory
from ss2r.rl.wrappers import SauteMDP


def make_saute(make_env, cfg: DictConfig) -> EnvironmentFactory:
    def make():
        env = make_env()
        env = SauteMDP(
            env,
            cfg.training.safety_budget,
            cfg.agent.safety_discount,
            cfg.saute.penalty,
        )
        return env

    return make


def make(cfg: DictConfig) -> EnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, task_config = get_domain_and_task(cfg)
    if "task" in task_config and (domain_name, task_config.task) in dm_control_envs:
        from ss2r.benchmark_suites.dm_control import make

        make_env = make(cfg)
    elif domain_name == "safe_adaptation_gym":
        from ss2r.benchmark_suites.safe_adaptation_gym import make

        make_env = make(cfg)
    else:
        raise NotImplementedError(f"Environment {domain_name} not implemented")
    if cfg.saute.enabled:
        make_env = make_saute(make_env, cfg)
    return make_env
