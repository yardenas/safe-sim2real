import random
from typing import Iterator, Optional

import jax
import jax.numpy as jnp

from ss2r.rl.trajectory import Transition


def pytrees_unstack(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees

def pytrees_stack(pytrees, axis=0):
    results = jax.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int, batch_size: int):
        self._random = random.Random(seed)
        self.capacity = capacity
        self.buffer: list[Optional[Transition]] = []
        self.position = 0
        self._batch_size = batch_size

    def store(self, transition: Transition):
        flat_transitions = pytrees_unstack(transition)
        for i in range(transition.reward.shape[0]):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = flat_transitions[i]
            self.position = int((self.position + 1) % self.capacity)

    def sample(self, num_samples: int) -> Iterator[Transition]:
        for _ in range(num_samples):
            batch = self._random.sample(self.buffer, self._batch_size)
            out = pytrees_stack(batch)
            yield out

    def __len__(self):
        return len(self.buffer)
