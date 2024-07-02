""" https://github.com/google-deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py """
from typing import Iterable, Iterator

import jax

from ss2r.rl.trajectory import TrajectoryData


def double_buffer(ds: Iterable[TrajectoryData]) -> Iterator[TrajectoryData]:
    """Keeps at least two batches on the accelerator.

    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.

    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.

    Args:
      ds: Iterable of batches of numpy arrays.

    Yields:
      Batches of sharded device arrays.
    """
    batch = None
    # TODO (yarden): should be sharded better
    # see equinox's docs for DDP
    devices = jax.local_devices()[0]
    for next_batch in ds:
        assert next_batch is not None
        next_batch = jax.device_put(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch
