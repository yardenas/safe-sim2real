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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import flax
import jax
import jax.numpy as jnp
from brax.training.types import Params


@flax.struct.dataclass
class SafePPONetworkParams:
    """Contains training state for the learner."""

    policy: Params
    value: Params
    cost_value: Params


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
):
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
      truncation: A float32 tensor of shape [T, B] with truncation signal.
      termination: A float32 tensor of shape [T, B] with termination signal.
      rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
      discount: TD discount.

    Returns:
      A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
      A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_t_plus_1 - values
    ) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def make_losses(
    ppo_network,
    clipping_epsilon,
    entropy_cost,
    reward_scaling,
    discounting,
    gae_lambda,
    normalize_advantage,
    cost_scaling,
    safety_budget,
    safety_discounting,
    safety_gae_lambda,
    use_ptsd,
    ptsd_lambda,
):
    def compute_policy_loss(
        policy_params,
        value_params,
        cost_value_params,
        normalizer_params,
        data,
        penalizer,
        penalizer_params,
        key,
    ):
        parametric_action_distribution = ppo_network.parametric_action_distribution
        policy_apply = ppo_network.policy_network.apply
        value_apply = ppo_network.value_network.apply
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        policy_logits = policy_apply(normalizer_params, policy_params, data.observation)
        baseline = value_apply(normalizer_params, value_params, data.observation)
        bootstrap_value = value_apply(
            normalizer_params, value_params, data.next_observation[-1]
        )
        rewards = data.reward * reward_scaling
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.discount) * (1 - truncation)
        target_log_probs = parametric_action_distribution.log_prob(
            policy_logits, data.extras["policy_extras"]["raw_action"]
        )
        behavior_log_probs = data.extras["policy_extras"]["log_prob"]
        _, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        rho_s = jnp.exp(target_log_probs - behavior_log_probs)
        surrogate1 = rho_s * advantages
        surrogate2 = (
            jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
        )
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, key))
        entropy_loss = -entropy_cost * entropy
        aux = {
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }
        if penalizer is not None:
            cost_value_apply = ppo_network.cost_value_network.apply
            cost = data.extras["state_extras"]["cost"] * cost_scaling
            if use_ptsd:
                cost += ptsd_lambda * data.extras["state_extras"]["disagreement"]
            cost_baseline = cost_value_apply(
                normalizer_params, cost_value_params, data.observation
            )
            cost_bootstrap_value = cost_value_apply(
                normalizer_params, cost_value_params, data.next_observation[-1]
            )
            vcs, cost_advantages = compute_gae(
                truncation=truncation,
                termination=termination,
                rewards=cost,
                values=cost_baseline,
                bootstrap_value=cost_bootstrap_value,
                lambda_=safety_gae_lambda,
                discount=safety_discounting,
            )
            cost_advantages -= cost_advantages.mean()
            cost_advantages *= rho_s
            cost_v_error = vcs - cost_baseline
            cost_v_loss = jnp.mean(cost_v_error * cost_v_error) * 0.5 * 0.5
            ongoing_costs = data.extras["state_extras"]["cumulative_cost"].max(0).mean()
            constraint = safety_budget - vcs.mean()
            # FIXME (YARDEN
            # )
            # policy_loss, penalizer_aux, _ = penalizer(
            #     policy_loss,
            #     constraint,
            #     jax.lax.stop_gradient(penalizer_params),
            #     rest=-cost_advantages.mean(),
            # )
            aux["constraint_estimate"] = constraint
            aux["cost_v_loss"] = cost_v_loss
            aux["ongoing_costs"] = ongoing_costs
            # aux |= penalizer_aux
        total_loss = policy_loss + entropy_loss
        return total_loss, aux

    def compute_value_loss(params, normalizer_params, data):
        value_apply = ppo_network.value_network.apply
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        baseline = value_apply(normalizer_params, params, data.observation)
        bootstrap_value = value_apply(
            normalizer_params, params, data.next_observation[-1]
        )
        rewards = data.reward * reward_scaling
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.discount) * (1 - truncation)
        vs, _ = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discounting,
        )
        v_error = vs - baseline
        v_loss = jnp.mean(v_error**2) * 0.25
        return v_loss, {"v_loss": v_loss}

    def compute_cost_value_loss(params, normalizer_params, data):
        cost_value_apply = ppo_network.cost_value_network.apply
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        cost = data.extras["state_extras"]["cost"] * cost_scaling
        if use_ptsd:
            cost += ptsd_lambda * data.extras["state_extras"]["disagreement"]
        cost_baseline = cost_value_apply(normalizer_params, params, data.observation)
        cost_bootstrap = cost_value_apply(
            normalizer_params, params, data.next_observation[-1]
        )
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.discount) * (1 - truncation)
        vcs, _ = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=cost,
            values=cost_baseline,
            bootstrap_value=cost_bootstrap,
            lambda_=safety_gae_lambda,
            discount=safety_discounting,
        )
        cost_error = vcs - cost_baseline
        cost_v_loss = jnp.mean(cost_error**2) * 0.25
        return cost_v_loss, {"cost_v_loss": cost_v_loss}

    return compute_policy_loss, compute_value_loss, compute_cost_value_loss
