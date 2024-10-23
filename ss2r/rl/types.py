from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, TypeAlias, Union

import jax
import numpy as np
from brax.training import types
from numpy import typing as npt
from omegaconf import DictConfig

FloatArray = Union[npt.NDArray[Union[np.float32, np.float64]], jax.Array]

SimulatorState = Any

Transition: TypeAlias = types.Transition


TrajectoryData = Transition


@dataclass
class Report:
    metrics: dict[str, float]
    videos: dict[str, npt.ArrayLike] = field(default_factory=dict)


class Simulator(Protocol):
    parallel_envs: int

    def rollout(
        self,
        policy: types.Policy,
        steps: int,
        seed: int,
        state: SimulatorState = None,
    ) -> tuple[SimulatorState, TrajectoryData]:
        ...

    def reset(self, seed: int | Sequence[int]) -> SimulatorState:
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
    def policy(self) -> types.Policy:
        ...

    def train(self, steps: int, simulator: Simulator) -> Report:
        ...
