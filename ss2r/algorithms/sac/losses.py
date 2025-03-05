# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.networks import SafeSACNetworks
from ss2r.algorithms.sac.penalizers import Penalizer
from ss2r.algorithms.sac.robustness import QTransformation

Transition: TypeAlias = types.Transition


def make_losses(
    sac_network: SafeSACNetworks,
    reward_scaling: float,
    discounting: float,
    safety_discounting: float,
    action_size: int,
    init_alpha: float | None,
):
    """Creates the SAC losses."""

    target_entropy = -0.5 * action_size if init_alpha is None else init_alpha
    policy_network = sac_network.policy_network
    qr_network = sac_network.qr_network
    qc_network = sac_network.qc_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        alpha_loss = jnp.mean(alpha_loss)
        return alpha_loss

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
        target_q_fn: QTransformation,
        safe: bool = False,
    ) -> jnp.ndarray:
        action = transitions.action
        q_network = qc_network if safe else qr_network
        gamma = safety_discounting if safe else discounting
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        key, another_key = jax.random.split(key)

        def policy(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_dist_params = policy_network.apply(
                normalizer_params, policy_params, obs
            )
            next_action = parametric_action_distribution.sample_no_postprocessing(
                next_dist_params, key
            )
            next_log_prob = parametric_action_distribution.log_prob(
                next_dist_params, next_action
            )
            next_action = parametric_action_distribution.postprocess(next_action)
            return next_action, next_log_prob

        q_fn = lambda obs, action: q_network.apply(
            normalizer_params, target_q_params, obs, action
        )
        target_q = target_q_fn(
            transitions,
            q_fn,
            policy,
            gamma,
            alpha,
            reward_scaling,
            another_key,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)
        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)
        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        qr_params: Params,
        qc_params: Params | None,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
        safety_budget: float,
        penalizer: Penalizer,
        penalizer_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        qr_action = qr_network.apply(
            normalizer_params, qr_params, transitions.observation, action
        )
        min_qr = jnp.min(qr_action, axis=-1)
        aux = {}
        actor_loss = -min_qr.mean()
        exploration_loss = (alpha * log_prob).mean()
        if qc_params is not None:
            assert qc_network is not None
            qc_action = qc_network.apply(
                normalizer_params, qc_params, transitions.observation, action
            )
            mean_qc = jnp.mean(qc_action, axis=-1)
            constraint = safety_budget - mean_qc.mean()
            actor_loss, penalizer_aux, penalizer_params = penalizer(
                actor_loss, constraint, jax.lax.stop_gradient(penalizer_params)
            )
            aux["constraint_estimate"] = constraint
            aux["cost"] = mean_qc.mean()
            aux["penalizer_params"] = penalizer_params
            aux |= penalizer_aux
        actor_loss += exploration_loss
        return actor_loss, aux

    return alpha_loss, critic_loss, actor_loss
