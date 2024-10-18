import random
from typing import Iterator, Optional

import numpy as np

from ssac.rl.trajectory import Transition


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, batch_size: int):
        self._random = random.Random(seed)
        self.capacity = capacity
        self.buffer: list[Optional[Transition]] = []
        self.position = 0
        self._batch_size = batch_size

    def store(self, transition: Transition):
        assert transition.cost.ndim == 1
        for i in range(transition.cost.shape[0]):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = Transition(*map(lambda x: x[i], transition))
            self.position = int((self.position + 1) % self.capacity)

    def sample(self, num_samples: int) -> Iterator[Transition]:
        for _ in range(num_samples):
            batch = self._random.sample(self.buffer, self._batch_size)
            out = Transition(*map(np.stack, zip(*batch)))  # type: ignore
            yield out

    def __len__(self):
        return len(self.buffer)
