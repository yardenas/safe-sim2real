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
from typing import Any, NamedTuple, TypeAlias

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.sac import networks as sac_networks
from brax.training.types import Params, PRNGKey

Transition: TypeAlias = types.Transition


def make_losses(
    sac_network: sac_networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
):
    """Creates the SAC losses."""

    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
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
        return jnp.mean(alpha_loss)

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        domain_params = transitions.extras.get("domain_parameters", None)
        if domain_params is not None:
            action = jnp.concatenate([transitions.action, domain_params], axis=-1)
        else:
            action = transitions.action
        q_old_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        next_dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.next_observation
        )
        next_action = parametric_action_distribution.sample_no_postprocessing(
            next_dist_params, key
        )
        next_log_prob = parametric_action_distribution.log_prob(
            next_dist_params, next_action
        )
        next_action = parametric_action_distribution.postprocess(next_action)
        if domain_params is not None:
            next_action = jnp.concatenate([next_action, domain_params], axis=-1)
        next_q = q_network.apply(
            normalizer_params,
            target_q_params,
            transitions.next_observation,
            next_action,
        )
        expand = lambda x: jnp.expand_dims(x, axis=-1)
        next_v = jnp.min(next_q, axis=-1) - alpha * expand(next_log_prob)
        if next_q.shape[1] > 1:
            cost = transitions.extras.get(
                "imagined_cost",
                transitions.extras.get("cost", jnp.zeros_like(transitions.reward)),
            )
            # FIXME (yarden): cost is not zeros
            cost = jnp.zeros_like(transitions.reward)
            reward = jnp.stack([transitions.reward, cost], axis=-1)
            # No need for exploration bonus in the constraints.
            next_v = next_v.at[:, 1].set(0)
        else:
            reward = expand(transitions.reward)
        target_q = jax.lax.stop_gradient(
            reward * reward_scaling
            + expand(transitions.discount) * discounting * next_v
        )
        q_error = q_old_action - expand(target_q)

        # Better bootstrapping for truncated episodes.
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= expand(expand(1 - truncation))

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
        safety_budget: float,
        lagrangian_params: LagrangianParams,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(
            normalizer_params, policy_params, transitions.observation
        )
        action = parametric_action_distribution.sample_no_postprocessing(
            dist_params, key
        )
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        domain_params = transitions.extras.get("domain_parameters", None)
        if domain_params is not None:
            action = jnp.concatenate([action, domain_params], axis=-1)
        q_action = q_network.apply(
            normalizer_params, q_params, transitions.observation, action
        )
        min_q = jnp.min(q_action, axis=-1)
        aux = {}
        if min_q.shape[1] > 1:
            q_r, q_c = jnp.split(min_q, 2, -1)
            constraint = safety_budget - q_c.mean()
            psi, cond = augmented_lagrangian(
                constraint,
                lagrangian_params.lagrange_multiplier,
                lagrangian_params.penalty_multiplier,
            )
            actor_loss = psi + jnp.mean(alpha * log_prob - q_r)
            aux["lagrangian_cond"] = cond
            aux["constraint_estimate"] = constraint
            aux["cost"] = q_c.mean()

        else:
            actor_loss = jnp.mean(alpha * log_prob - min_q)
        return actor_loss, aux

    return alpha_loss, critic_loss, actor_loss


class LagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    penalty_multiplier: jax.Array


def augmented_lagrangian(
    constraint: jax.Array,
    lagrange_multiplier: jax.Array,
    penalty_multiplier: jax.Array,
) -> jax.Array:
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    g = -constraint
    c = penalty_multiplier
    cond = lagrange_multiplier + c * g
    psi = jnp.where(
        jnp.greater(cond, 0.0),
        lagrange_multiplier * g + c / 2.0 * g**2,
        -1.0 / (2.0 * c) * lagrange_multiplier**2,
    )
    return psi, cond


def update_augmented_lagrangian(
    cond: jax.Array, penalty_multiplier: jax.Array, penalty_multiplier_factor: float
):
    new_penalty_multiplier = jnp.clip(
        penalty_multiplier * (1.0 + penalty_multiplier_factor), penalty_multiplier, 1.0
    )
    new_lagrange_multiplier = jnn.relu(cond)
    return LagrangianParams(new_lagrange_multiplier, new_penalty_multiplier)
