from typing import Any, Callable, Tuple, TypeAlias

import flax
import jax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import Params

from ss2r.algorithms.ppo import losses as ppo_losses

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.SafePPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    penalizer_params: Params
    env_steps: jnp.ndarray


Metrics: TypeAlias = types.Metrics


TrainingStep: TypeAlias = Callable[
    [
        Tuple[TrainingState, envs.State, types.PRNGKey, int],
    ],
    Tuple[Tuple[TrainingState, envs.State, types.PRNGKey], Metrics],
]

TrainingStepFactory: TypeAlias = Callable[
    [
        Callable[[Any], jax.Array],
        optax.GradientTransformation,
        envs.Env,
        int,
        int,
        Callable[[Any], types.Policy],
        int,
        int,
        int,
        int,
        bool,
        bool,
    ],
    TrainingStep,
]


InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]
