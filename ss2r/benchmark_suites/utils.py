from omegaconf import DictConfig


def get_domain_name(cfg: DictConfig) -> str:
    return cfg.environment.domain_name


def get_task_config(cfg: DictConfig) -> DictConfig:
    return cfg.environment
