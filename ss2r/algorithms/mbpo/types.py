import flax
import jax.numpy as jnp
import optax
from brax.training.acme import running_statistics
from brax.training.types import Params


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    behavior_policy_optimizer_state: optax.OptState
    behavior_policy_params: Params
    backup_policy_params: Params
    behavior_qr_optimizer_state: optax.OptState
    behavior_qr_params: Params
    behavior_qc_optimizer_state: optax.OptState | None
    behavior_qc_params: Params | None
    backup_qc_optimizer_state: optax.OptState | None
    backup_qc_params: Params | None
    model_params: Params
    model_optimizer_state: optax.OptState
    behavior_target_qr_params: Params
    behavior_target_qc_params: Params | None
    backup_target_qc_params: Params | None
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    penalizer_params: Params
