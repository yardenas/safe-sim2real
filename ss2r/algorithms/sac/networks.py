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

from typing import Protocol, Sequence, TypeAlias, TypeVar

from brax.training import distribution, networks, types
import brax.training.agents.sac.networks as sac_networks
from flax import linen

make_inference_fn = sac_networks.make_inference_fn
SACNetworks: TypeAlias = sac_networks.SACNetworks
NetworkType = TypeVar('NetworkType', covariant=True)

class DomainRandomizationNetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        *,
        domain_randomization_size: int = 0,
    ) -> NetworkType:
        pass


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    *,
    domain_randomization_size: int = 0,
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
    q_network = networks.make_q_network(
        observation_size,
        action_size + domain_randomization_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )
