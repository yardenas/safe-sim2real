from omegaconf import DictConfig

from ss2r.benchmark_suites.utils import get_domain_and_task
from ss2r.rl.types import SimulatorFactory


def make(cfg: DictConfig) -> SimulatorFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, _ = get_domain_and_task(cfg)
    if domain_name == "brax":
        from ss2r.benchmark_suites.brax import make

        make_env = make(cfg)
    else:
        raise NotImplementedError(f"Environment {domain_name} not implemented")
    return make_env
