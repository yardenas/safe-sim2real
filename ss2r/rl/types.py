from dataclasses import dataclass, field
from typing import Callable, Protocol, Union

import jax
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import typing as npt
from omegaconf import DictConfig

from ss2r.rl.epoch_summary import EpochSummary
from ss2r.rl.trajectory import TrajectoryData

FloatArray = npt.NDArray[Union[np.float32, np.float64]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

Policy = Union[Callable[[jax.Array, jax.Array | None], jax.Array], jax.Array]


@dataclass
class Report:
    metrics: dict[str, float]
    videos: dict[str, npt.ArrayLike] = field(default_factory=dict)


class Agent(Protocol):
    config: DictConfig

    def __call__(self, observation: FloatArray, train: bool = False) -> FloatArray:
        ...

    def observe(self, trajectory: TrajectoryData) -> None:
        ...

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        ...
