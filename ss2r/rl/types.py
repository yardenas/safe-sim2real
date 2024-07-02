from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Protocol, Union

import jax
import numpy as np
from brax.training.types import Policy
from numpy import typing as npt
from omegaconf import DictConfig


FloatArray = Union[npt.NDArray[Union[np.float32, np.float64]], jax.Array]


class Transition(NamedTuple):
    observation: FloatArray
    next_observation: FloatArray
    action: FloatArray
    reward: FloatArray
    cost: FloatArray
    discount: FloatArray
    extras: FloatArray = ()


TrajectoryData = Transition


@dataclass
class Report:
    metrics: dict[str, float]
    videos: dict[str, npt.ArrayLike] = field(default_factory=dict)


class Simulator(Protocol):
    action_size: int
    observation_size: int

    def rollout(self, policy: Policy, steps: int) -> TrajectoryData:
        ...


SimulatorFactory = Callable[[], Simulator]


class Agent(Protocol):
    config: DictConfig

    @property
    def policy(self) -> Policy:
        ...

    def train(self, steps: int, simulator: Simulator) -> Report:
        ...
