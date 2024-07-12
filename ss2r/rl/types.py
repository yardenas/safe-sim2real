from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Protocol, Sequence, Union

import jax
import numpy as np
from brax.training.types import Policy
from numpy import typing as npt
from omegaconf import DictConfig


FloatArray = Union[npt.NDArray[Union[np.float32, np.float64]], jax.Array]

SimulatorState = Any


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
    def rollout(
        self,
        policy: Policy,
        steps: int,
        seed: int | Sequence[int],
        state: SimulatorState,
    ) -> tuple[SimulatorState, TrajectoryData]:
        ...

    def reset(self, seed: int | Sequence[int]) -> TrajectoryData:
        ...

    @property
    def action_size(self) -> int:
        ...

    @property
    def observation_size(self) -> int:
        ...


SimulatorFactory = Callable[[], Simulator]


class Agent(Protocol):
    config: DictConfig

    @property
    def policy(self) -> Policy:
        ...

    def train(self, steps: int, simulator: Simulator) -> Report:
        ...
