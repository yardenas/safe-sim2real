
from omegaconf import DictConfig


def get_domain_and_task(cfg: DictConfig) -> tuple[str, DictConfig]:
    assert len(cfg.environment.keys()) == 1
    domain_name, task = list(cfg.environment.items())[0]
    return domain_name, task
