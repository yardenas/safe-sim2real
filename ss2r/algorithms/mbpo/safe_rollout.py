from typing import Any, Callable, Tuple

import brax.training.agents.sac.networks as sac_networks
import jax.numpy as jnp
from brax.training import types
from brax.training.types import PolicyParams, PRNGKey

from ss2r.algorithms.mbpo.networks import MBPONetworks
from ss2r.algorithms.mbpo.types import TrainingState


def get_inference_policy_params(safe: bool, safety_budget=float("inf")) -> Any:
    def get_params(training_state: TrainingState) -> Any:
        if safe:
            return (
                training_state.behavior_policy_params,
                training_state.backup_qc_params,
                safety_budget,
            )
        else:
            return training_state.behavior_policy_params

    return get_params


def make_safe_inference_fn(
    mbpo_networks: MBPONetworks,
    inital_backup_policy_params,
    initial_normalizer_params,
    scaling_fn,
) -> Callable[[Any, bool], types.Policy]:
    """Creates params and inference function for the SAC agent."""
    backup_policy = sac_networks.make_inference_fn(mbpo_networks)(
        (initial_normalizer_params, inital_backup_policy_params), deterministic=True
    )

    def make_policy(
        params: PolicyParams,
        deterministic: bool = False,
    ) -> types.Policy:
        (
            normalizer_params,
            (policy_params, qc_params, safety_budget),
        ) = params

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = mbpo_networks.policy_network.apply(
                normalizer_params, policy_params, observations
            )

            mode_a = mbpo_networks.parametric_action_distribution.mode(logits)
            if deterministic:
                behavioral_action = mode_a
            else:
                behavioral_action = mbpo_networks.parametric_action_distribution.sample(
                    logits, key_sample
                )
            if mbpo_networks.qc_network is not None:
                qc = mbpo_networks.qc_network.apply(
                    normalizer_params, qc_params, observations, behavioral_action
                ).mean(axis=-1)
            else:
                raise ValueError("QC network is not defined, cannot do shielding.")
            accumulated_cost = observations["cumulative_cost"][:, 0]
            expected_total_cost = accumulated_cost + scaling_fn(qc)
            backup_action = backup_policy(observations, key_sample)[0]
            safe = expected_total_cost[:, None] < safety_budget
            safe_action = jnp.where(
                safe,
                behavioral_action,
                backup_action,
            )
            extras = {
                "intervention": 1 - safe[:, 0].astype(jnp.float32),
                "policy_distance": jnp.linalg.norm(mode_a - safe_action, axis=-1),
                "safety_gap": jnp.maximum(
                    expected_total_cost - safety_budget,
                    jnp.zeros_like(expected_total_cost),
                ),
                "expected_total_cost": expected_total_cost,
            }
            return safe_action, extras

        return policy

    return make_policy
