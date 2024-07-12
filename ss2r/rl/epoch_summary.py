from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
from numpy import typing as npt

from ss2r.rl.types import TrajectoryData


@dataclass
class EpochSummary:
    _data: list[TrajectoryData] = field(default_factory=list)
    cost_boundary: float = 25.0

    @property
    def empty(self):
        return len(self._data) == 0

    @property
    def metrics(self) -> Tuple[float, float, float]:
        rewards, costs = [], []
        for trajectory_batch in self._data:
            rewards.append(trajectory_batch.reward)
            costs.append(trajectory_batch.cost)
        # Stack data from all tasks on the first axis,
        # giving a [#tasks, #episodes, #time, ...] shape.
        stacked_rewards = np.stack(rewards)
        stacked_costs = np.stack(costs)
        return (
            _objective(stacked_rewards),
            _cost_rate(stacked_costs),
            _feasibility(stacked_costs, self.cost_boundary),
        )

    @property
    def videos(self):
        all_vids = []
        for trajectory_batch in self._data:
            for trajectory in trajectory_batch:
                if len(trajectory.frames) > 0:
                    all_vids.append(trajectory.frames)
        if len(all_vids) == 0:
            return None
        vids = np.asarray(all_vids)[-1].transpose(1, 0, -1, 2, 3)
        return vids

    def extend(self, samples: TrajectoryData) -> None:
        self._data.append(samples)


def _objective(rewards: npt.NDArray[Any]) -> float:
    return float(rewards.sum(1).mean())


def _cost_rate(costs: npt.NDArray[Any]) -> float:
    return float(costs.mean())


def _feasibility(costs: npt.NDArray[Any], boundary: float) -> float:
    return float((costs.sum(2).mean(1) <= boundary).mean())
