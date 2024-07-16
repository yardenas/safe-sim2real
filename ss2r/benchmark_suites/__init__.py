from omegaconf import DictConfig

from ss2r.benchmark_suites.utils import get_domain_name
from ss2r.rl.types import SimulatorFactory


def make(cfg: DictConfig) -> SimulatorFactory:
    assert len(cfg.environment.keys()) == 2
    domain_name = get_domain_name(cfg)
    if domain_name == "brax":
        from ss2r.benchmark_suites.brax import make

        make_env = make(cfg)
    else:
        raise NotImplementedError(f"Environment {domain_name} not implemented")
    return make_env
