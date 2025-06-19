from typing import Generic, Sequence, Tuple, TypeVar

from brax.training.agents.sac import checkpoint
from absl import logging
import flax
import jax
import jax.numpy as jnp
from brax.training.replay_buffers import (
    ReplayBuffer,
    ReplayBufferState,
    UniformSamplingQueue,
)

from ss2r.common.wandb import get_wandb_checkpoint

Sample = TypeVar("Sample")


@flax.struct.dataclass
class RAEReplayBufferState:
    offline_state: ReplayBufferState
    online_state: ReplayBufferState
    key: jax.Array


class RAEReplayBuffer(ReplayBuffer[RAEReplayBufferState, Sample], Generic[Sample]):
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
        wandb_ids: Sequence[str],
    ):
        self.sample_batch_size = sample_batch_size
        self.online_sample_size = sample_batch_size // 2
        self.offline_sample_size = sample_batch_size - self.online_sample_size
        self.online_buffer = UniformSamplingQueue(
            max_replay_size, dummy_data_sample, self.online_sample_size
        )
        all_data = prepare_offline_data(wandb_ids)
        max_size = all_data.shape[0]
        self.offline_buffer = UniformSamplingQueue(
            max_size,
            dummy_data_sample,
            self.offline_sample_size,
        )
        logging.info("Restoring replay buffer state with %d samples", max_size)
        self._unflattened_data = self.offline_buffer._unflatten_fn(all_data)

    def init(self, key: jax.Array) -> RAEReplayBufferState:
        """Initialize both buffers and load offline data from disk."""
        key_online, key_offline, key_next = jax.random.split(key, 3)
        online_state = self.online_buffer.init(key_online)
        offline_state = self.offline_buffer.init(key_offline)
        offline_state = self.offline_buffer.insert(
            offline_state, self._unflattened_data
        )
        return RAEReplayBufferState(
            online_state=online_state,  # type: ignore
            offline_state=offline_state,  # type: ignore
            key=key_next,
        )

    def insert_internal(
        self, state: RAEReplayBufferState, samples: Sample
    ) -> RAEReplayBufferState:
        """Insert new online samples."""
        new_online_state = self.online_buffer.insert(state.online_state, samples)
        new_offline_state = state.offline_state
        return state.replace(  # type: ignore
            online_state=new_online_state,
            offline_state=new_offline_state,
        )

    def sample_internal(
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

    def size(self, state: RAEReplayBufferState) -> int:
        return self.online_buffer.size(state.online_state) + self.offline_buffer.size(
            state.offline_state
        )


def prepare_offline_data(wandb_ids):
    data = []
    for wandb_id in wandb_ids:
        checkpoint_path = get_wandb_checkpoint(wandb_id)
        params = checkpoint.load(checkpoint_path)
        assert len(params) == 9
        replay_buffer_state = params[-1]
        data.append(_find_first_nonzeros(replay_buffer_state.data))
    concatenated_data = jnp.concatenate(data, axis=0)
    return concatenated_data


def _find_first_nonzeros(x):
    zero_row_mask = jnp.all(x == 0, axis=1)
    first_zero_row_index = jnp.argmax(zero_row_mask)
    return x[:first_zero_row_index]
