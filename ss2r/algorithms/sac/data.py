from typing import Tuple

from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Policy, PRNGKey

from ss2r.algorithms.sac import ReplayBufferState, float16


def collect_single_step(
    env: envs.Env,
    policy: Policy,
    normalizer_params: running_statistics.RunningStatisticsState,
    replay_buffer: ReplayBuffer,
    env_state: envs.State,
    buffer_state: ReplayBufferState,
    key: PRNGKey,
    extra_fields: Tuple[str, ...] = ("truncation",),
) -> Tuple[
    running_statistics.RunningStatisticsState,
    envs.State,
    ReplayBufferState,
]:
    # TODO (yarden): if I ever need to sample states based on value functions
    # one way to code it is to add a function to the StatePropagation wrapper
    # that receives a function that takes states and returns their corresponding value functions
    env_state, transitions = acting.actor_step(
        env, env_state, policy, key, extra_fields=extra_fields
    )
    normalizer_params = running_statistics.update(
        normalizer_params, transitions.observation
    )
    buffer_state = replay_buffer.insert(buffer_state, float16(transitions))
    return normalizer_params, env_state, buffer_state
