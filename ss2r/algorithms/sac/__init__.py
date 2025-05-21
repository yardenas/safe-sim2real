import functools
from typing import Any, Callable, Protocol, Sequence, Tuple, TypeAlias

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, Policy, PolicyParams, PRNGKey

Metrics: TypeAlias = types.Metrics
Transition: TypeAlias = types.Transition
InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState: TypeAlias = Any

make_float = lambda x, t: jax.tree.map(lambda y: y.astype(t), x)
float16 = functools.partial(make_float, t=jnp.float16)
float32 = functools.partial(make_float, t=jnp.float32)


class MakePolicyFn(Protocol):
    def __call__(self, policy_params: PolicyParams) -> Policy:
        ...


CollectDataFn = Callable[
    [
        envs.Env,
        MakePolicyFn,
        Params,
        running_statistics.RunningStatisticsState,
        ReplayBuffer,
        envs.State,
        ReplayBufferState,
        PRNGKey,
        Tuple[str, ...],
    ],
    Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ],
]


class UnrollFn(Protocol):
    def __call__(
        self,
        env: envs.Env,
        env_state: envs.State,
        make_policy_fn: MakePolicyFn,
        policy_params: PolicyParams,
        key: PRNGKey,
        *,
        extra_fields: Sequence[str],
    ) -> Tuple[envs.State, Transition]:
        ...
