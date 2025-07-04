from typing import Mapping, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training.replay_buffers import (
    ReplayBuffer,
    ReplayBufferState,
)
from brax.training.types import PRNGKey, Transition


@flax.struct.dataclass
class PytreeReplayBufferState:
    """Contains data related to a replay buffer."""

    data: Transition
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: PRNGKey


class PytreeUniformSamplingQueue(ReplayBuffer[PytreeReplayBufferState, Transition]):
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Transition,
        sample_batch_size: int,
        *,
        store_pixels_in_cpu: bool = True,
    ):
        # Create per-field data arrays with shapes [max_replay_size, ...]
        self._data_template = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(
                shape=(max_replay_size,) + x.shape,
                dtype=x.dtype,
            ),
            dummy_data_sample,
        )
        self._max_replay_size = max_replay_size
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.store_pixels_in_cpu = store_pixels_in_cpu

    def init(self, key: jax.random.PRNGKey) -> PytreeReplayBufferState:
        data = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, x.dtype), self._data_template
        )
        if self.store_pixels_in_cpu:
            cpu = jax.devices("cpu")[0]
            data = data._replace(
                observation=move_pixels_to_device(data.observation, cpu),
                next_observation=move_pixels_to_device(data.next_observation, cpu),
            )
        return PytreeReplayBufferState(  # type: ignore
            data=data,
            insert_position=jnp.zeros((), jnp.int32),
            sample_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert_internal(
        self, buffer_state: PytreeReplayBufferState, samples: Transition
    ) -> PytreeReplayBufferState:
        """Insert samples into buffer at the current insert_position."""

        data = buffer_state.data
        position = buffer_state.insert_position
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data.reward) - position - len(samples.reward))

        def roll_one_field(field_data):
            return jnp.roll(field_data, roll, axis=0)

        data = jax.lax.cond(
            roll,
            lambda: jax.tree_util.tree_map(roll_one_field, data),
            lambda: data,
        )
        position = position + roll

        def insert_one_field(field_data, field_sample):
            return jax.lax.dynamic_update_slice_in_dim(
                field_data, field_sample, position, axis=0
            )

        # Insert data at position
        if self.store_pixels_in_cpu:
            cpu = jax.devices("cpu")[0]
            samples = samples._replace(
                observation=move_pixels_to_device(samples.observation, cpu),
                next_observation=move_pixels_to_device(samples.next_observation, cpu),
            )
        new_data = jax.tree_util.tree_map(insert_one_field, data, samples)
        position = (position + len(samples.reward)) % (len(data.reward) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)
        return buffer_state.replace(  # type: ignore
            data=new_data,
            insert_position=position,
            sample_position=sample_position,
        )

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )  # pytype: disable=bad-return-type  # jax-ndarray

    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Transition]:
        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )

        # Sample each field from the buffer using gathered indices
        def sample_field(field_data):
            return jnp.take(field_data, idx, axis=0, mode="wrap")

        batch = jax.tree_util.tree_map(sample_field, buffer_state.data)
        if self.store_pixels_in_cpu:
            gpus = jax.devices("gpu")
            if not gpus:
                device = jax.devices("cpu")[0]
            else:
                device = gpus[0]
            batch = batch._replace(
                observation=move_pixels_to_device(batch.observation, device),
                next_observation=move_pixels_to_device(batch.next_observation, device),
            )
        return buffer_state.replace(key=key), batch


def move_pixels_to_device(
    observation: Mapping[str, jax.Array] | jax.Array,
    to_device: jax.Device,
) -> Mapping[str, jax.Array] | jax.Array:
    """Move values to CPU if their key matches any regex pattern."""
    if isinstance(observation, jax.Array):
        return observation
    return {
        k: jax.device_put(v, to_device) if k.startswith("pixels/") else v
        for k, v in observation.items()
    }
