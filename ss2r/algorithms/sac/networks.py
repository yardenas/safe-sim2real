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

from typing import Any, Callable, Sequence, TypeVar

import brax.training.agents.sac.networks as sac_networks
import flax
import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

make_inference_fn = sac_networks.make_inference_fn
NetworkType = TypeVar("NetworkType", covariant=True)


@flax.struct.dataclass
class SafeSACNetworks:
    policy_network: networks.FeedForwardNetwork
    qr_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution
    qc_network: networks.FeedForwardNetwork | None


class BroNet(linen.Module):
    layer_sizes: Sequence[int]
    activation: Callable
    kernel_init: Callable = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        assert all(size == self.layer_sizes[0] for size in self.layer_sizes[:-1])
        x = linen.Dense(features=self.layer_sizes[0], kernel_init=self.kernel_init)(x)
        x = linen.LayerNorm()(x)
        x = self.activation(x)
        for _ in range(len(self.layer_sizes)):
            residual = x
            x = linen.Dense(features=self.layer_sizes[0], kernel_init=self.kernel_init)(
                x
            )
            x = linen.LayerNorm()(x)
            x = self.activation(x)
            x = linen.Dense(features=self.layer_sizes[0], kernel_init=self.kernel_init)(
                x
            )
            x = linen.LayerNorm()(x)
            x += residual
        return x


class MLP(linen.Module):
    layer_sizes: Sequence[int]
    activation: Callable
    kernel_init: Callable = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, x):
        for i, size in enumerate(self.layer_sizes):
            x = linen.Dense(features=size, kernel_init=self.kernel_init)(x)
            if i < len(self.layer_sizes) - 1:
                x = linen.LayerNorm()(x)
                x = self.activation(x)
        return x


def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
    use_bro: bool = True,
) -> networks.FeedForwardNetwork:
    """Creates a value network."""

    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            net = BroNet if use_bro else MLP
            for _ in range(self.n_critics):
                q = net(  # type: ignore
                    layer_sizes=list(hidden_layer_sizes) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    use_bro: bool = True,
    *,
    domain_randomization_size: int = 0,
    safe: bool = False,
) -> SafeSACNetworks:
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
    qr_network = make_q_network(
        observation_size,
        action_size + domain_randomization_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
    )
    if safe:
        qc_network = make_q_network(
            observation_size,
            action_size + domain_randomization_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        old_apply = qc_network.apply
        qc_network.apply = lambda *args, **kwargs: jnn.softplus(
            old_apply(*args, **kwargs)
        )
    else:
        qc_network = None
    return SafeSACNetworks(
        policy_network=policy_network,
        qr_network=qr_network,
        qc_network=qc_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
