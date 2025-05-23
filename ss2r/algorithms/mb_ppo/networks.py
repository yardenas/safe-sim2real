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

"""PPO networks."""

from typing import Sequence, Tuple, Callable, Dict, Any

import flax
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey
from flax import linen
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
from brax.training.networks import MLP
from brax.training.networks import FeedForwardNetwork
from brax.training.networks import ActivationFn
from brax.training.networks import _get_obs_state_size



@flax.struct.dataclass
class MBPPONetworks:
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


def make_value_network(
    obs_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    obs_key: str = 'state',
    use_bro: bool = True,
) -> FeedForwardNetwork:
  """Creates a value network."""
  net = BroNet if use_bro else MLP
  value_module = net(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform(),
  )

  def apply(processor_params, value_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
    return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

  obs_size = _get_obs_state_size(obs_size, obs_key)
  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply
  )


def make_world_model_ensemble(
    obs_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_ensemble: int = 5,
    obs_key: str = "state",
    use_bro: bool = True,
) -> networks.FeedForwardNetwork:
    """Creates a model network."""

    class MModule(linen.Module):
        """M Module."""

        n_ensemble: int
        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            net = BroNet if use_bro else MLP
            for _ in range(self.n_ensemble):
                m = net(  # type: ignore
                    hidden_dims=list(hidden_layer_sizes) + [obs_size],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(m)
            return jnp.concatenate(res, axis=-1)
        
    obs_size = _get_obs_state_size(obs_size, obs_key)
    q_module = MModule(n_ensemble=n_ensemble, obs_size=obs_size)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[obs_key]
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


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


def make_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    use_bro: bool = True,
    n_ensemble: int = 5,
) -> MBPPONetworks:
    """Make PPO networks with preprocessor."""
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
        use_bro=use_bro,
    )
    cost_value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        use_bro=use_bro,
    )

    return MBPPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        cost_value_network=cost_value_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
