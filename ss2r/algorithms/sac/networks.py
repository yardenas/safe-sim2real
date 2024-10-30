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

"""SAC networks."""

from typing import Any, Callable, Protocol, Sequence, TypeAlias, TypeVar

import brax.training.agents.sac.networks as sac_networks
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from brax.training.networks import MLP, FeedForwardNetwork
from flax import linen

make_inference_fn = sac_networks.make_inference_fn
SACNetworks: TypeAlias = sac_networks.SACNetworks
NetworkType = TypeVar("NetworkType", covariant=True)

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class DomainRandomizationNetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        *,
        domain_randomization_size: int = 0,
        safe: bool = False,
    ) -> NetworkType:
        pass


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    output_size: int = 1,
) -> FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(hidden_layer_sizes) + [output_size],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(q)
            return jnp.stack(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    *,
    domain_randomization_size: int = 0,
    safe: bool = False,
) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    q_network = make_q_network(
        observation_size,
        action_size + domain_randomization_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        output_size=int(safe) + 1,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )
