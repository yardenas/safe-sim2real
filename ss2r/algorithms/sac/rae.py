from typing import Generic, Tuple, TypeVar

import flax
import jax
import jax.numpy as jnp
from absl import logging
from brax.training.replay_buffers import (
    ReplayBuffer,
    ReplayBufferState,
    UniformSamplingQueue,
)

Sample = TypeVar("Sample")


@flax.struct.dataclass
class RAEReplayBufferState:
    offline_state: ReplayBufferState
    online_state: ReplayBufferState
    key: jax.Array


class RAEReplayBuffer(ReplayBuffer[RAEReplayBufferState, Sample], Generic[Sample]):
    """Users must set offline_data_state to load offline data from disk."""

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
    ):
        self.sample_batch_size = sample_batch_size
        self.online_sample_size = sample_batch_size // 2
        self.offline_sample_size = sample_batch_size - self.online_sample_size
        self.offline_data_state = None
        self.online_buffer = UniformSamplingQueue(
            max_replay_size, dummy_data_sample, self.online_sample_size
        )
        self.offline_buffer = UniformSamplingQueue(
            max_replay_size, dummy_data_sample, self.offline_sample_size
        )

    def init(self, key: jax.Array) -> RAEReplayBufferState:
        """Initialize both buffers and load offline data from disk."""
        key_online, key_offline, key_next = jax.random.split(key, 3)
        online_state = self.online_buffer.init(key_online)
        if self.offline_data_state is None:
            logging.warn("No offline data found, initializing from online data.")
            self.offline_data_state = self.offline_buffer.init(key_offline)
        return RAEReplayBufferState(
            online_state=online_state,  # type: ignore
            offline_state=self.offline_data_state,  # type: ignore
            key=key_next,
        )

    def insert(
        self, state: RAEReplayBufferState, samples: Sample
    ) -> RAEReplayBufferState:
        """Insert new online samples."""
        new_online_state = self.online_buffer.insert(state.online_state, samples)
        if self.offline_data_state is not None:
            self.offline_buffer.insert(state.offline_state, samples)
        return state.replace(online_state=new_online_state)  # type: ignore

    def sample(
        self, state: RAEReplayBufferState
    ) -> Tuple[RAEReplayBufferState, Sample]:
        """Sample from both buffers and return merged batch."""
        key_online, key_offline, new_key = jax.random.split(state.key, 3)
        online_state, online_samples = self.online_buffer.sample(
            state.online_state.replace(key=key_online)
        )
        offline_state, offline_samples = self.offline_buffer.sample(
            state.offline_state.replace(key=key_offline)
        )
        combined_samples = jax.tree_util.tree_map(
            lambda o, f: jnp.concatenate([o, f], axis=0),
            online_samples,
            offline_samples,
        )
        new_state = state.replace(  # type: ignore
            online_state=online_state,
            offline_state=offline_state,
            key=new_key,
        )
        return new_state, combined_samples
