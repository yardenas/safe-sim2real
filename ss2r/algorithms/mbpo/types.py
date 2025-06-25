import flax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics
from brax.training.types import Params


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    backup_policy_params: Params
    qr_optimizer_state: optax.OptState
    qr_params: Params
    qc_optimizer_state: optax.OptState | None
    qc_params: Params | None
    behavior_qc_optimizer_state: optax.OptState | None
    behavior_qc_params: Params | None
    model_params: Params
    model_optimizer_state: optax.OptState
    target_qr_params: Params
    target_qc_params: Params | None
    behavior_target_qc_params: Params | None
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    penalizer_params: Params
