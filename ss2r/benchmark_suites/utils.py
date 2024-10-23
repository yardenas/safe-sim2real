from omegaconf import DictConfig


def get_task_config(cfg: DictConfig) -> DictConfig:
    return list(cfg.environment.values())[1]
