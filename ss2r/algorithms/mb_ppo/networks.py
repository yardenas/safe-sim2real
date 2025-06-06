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

"""ModelBasedPPO networks."""

from typing import Callable, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen


@flax.struct.dataclass
class MBPPONetworks:
    model_network: networks.FeedForwardNetwork
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    cost_value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


class BroNet(linen.Module):
    hidden_dims: Sequence[int]
    activation: Callable
    kernel_init: Callable = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        assert all(size == self.hidden_dims[0] for size in self.hidden_dims[:-1])
        x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
        x = linen.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(len(self.hidden_dims) - 1):
            residual = x
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(
                x
            )
            x = linen.LayerNorm()(x)
            x = self.activation(x)
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(
                x
            )
            x = linen.LayerNorm()(x)
            x += residual
        x = linen.Dense(self.hidden_dims[-1], kernel_init=self.kernel_init)(x)
        return x


def make_inference_fn(ppo_networks: MBPPONetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            param_subset = (params[0], params[1])  # normalizer and policy params
            logits = policy_network.apply(*param_subset, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def make_world_model_ensemble(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    n_ensemble: int = 5,
    use_bro: bool = True,
    learn_std: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a model network."""

    # Convert obs_size to integer if it's a shape tuple
    if isinstance(obs_size, (tuple, list)):
        obs_size = obs_size[0] if len(obs_size) == 1 else sum(obs_size)
    obs_size = int(obs_size)

    class MModule(linen.Module):
        """M Module."""

        n_ensemble: int
        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            net = BroNet if use_bro else networks.MLP
            single_output_dim = obs_size + 2  # +2 for reward and cost
            if learn_std:
                single_output_dim *= 2
            output_dim = single_output_dim
            hidden_dims = list(hidden_layer_sizes) + [output_dim]
            for _ in range(self.n_ensemble):
                m = net(
                    hidden_dims=hidden_dims,
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(m)
            ensemble_output = jnp.stack(res, 0)
            return ensemble_output

    model = MModule(n_ensemble=n_ensemble, obs_size=obs_size)

    def apply(processor_params, params, obs, actions):
        obs_processed = preprocess_observations_fn(obs, processor_params)
        raw_output = model.apply(params, obs_processed, actions)
        # Std devs also need to match the shape (B, E, feature_dim)
        if not learn_std:
            obs, reward, cost = (
                raw_output[..., :obs_size],
                raw_output[..., obs_size],
                raw_output[..., obs_size + 1],
            )
            obs_std = jnp.ones_like(obs) * 1e-3
            reward_std = jnp.ones_like(reward) * 1e-3
            cost_std = jnp.ones_like(cost) * 1e-3
        else:
            means, stds = jnp.split(raw_output, 2, axis=-1)
            obs, reward, cost = (
                means[..., :obs_size],
                means[..., obs_size],
                means[..., obs_size + 1],
            )
            obs_std, reward_std, cost_std = (
                stds[..., :obs_size],
                stds[..., obs_size],
                stds[..., obs_size + 1],
            )
        return (obs, reward, cost), (obs_std, reward_std, cost_std)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_mb_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    model_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    n_ensemble: int = 5,
    model_use_bro: bool = True,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    learn_std: bool = False,
) -> MBPPONetworks:
    """Make PPO networks with preprocessor."""

    # Convert observation_size to integer if it's a shape tuple
    if isinstance(observation_size, (tuple, list)):
        observation_size = (
            observation_size[0] if len(observation_size) == 1 else sum(observation_size)
        )
    observation_size = int(observation_size)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        obs_key=policy_obs_key,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )
    cost_value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )
    model_network = make_world_model_ensemble(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=model_hidden_layer_sizes,
        activation=activation,
        n_ensemble=n_ensemble,
        use_bro=model_use_bro,
        learn_std=learn_std,
    )

    return MBPPONetworks(
        model_network=model_network,
        policy_network=policy_network,
        value_network=value_network,
        cost_value_network=cost_value_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
