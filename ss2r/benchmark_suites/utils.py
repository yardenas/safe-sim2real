
from omegaconf import DictConfig


def get_domain_name(cfg: DictConfig) -> str:
    return list(cfg.environment.keys())[0]

def get_task_config(cfg: DictConfig) -> DictConfig:
    return list(cfg.environment.values())[1]
