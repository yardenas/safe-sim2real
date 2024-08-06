from typing import Protocol

import jax
from omegaconf import DictConfig
from brax.training.types import PRNGKey


class StateSampler(Protocol):
    def __init__(self, cfg: DictConfig, state_dim: int) -> None:
        ...

    def sample(self, rng: PRNGKey):
        ...


class UniformStateSampler(StateSampler):
    def __init__(self, cfg: DictConfig, state_dim: int) -> None:
        self.state_dim = state_dim

    def sample(self, rng: PRNGKey):
        return jax.random.uniform(rng, shape=(self.state_dim,), minval=-1.0, maxval=1.0)
