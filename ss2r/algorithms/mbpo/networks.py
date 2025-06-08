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

from typing import Any, Callable, Protocol, Sequence, TypeVar

import brax.training.agents.sac.networks as sac_networks
import flax
import jax
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.sac.networks import make_q_network

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

make_inference_fn = sac_networks.make_inference_fn
NetworkType = TypeVar("NetworkType", covariant=True)


class NetworkFactory(Protocol[NetworkType]):
    def __call__(
        self,
        observation_size: int,
        action_size: int,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        *,
        n_critics: int = 2,
        n_heads: int = 1,
        use_bro: bool = True,
    ) -> NetworkType:
        pass


@flax.struct.dataclass
class MBPONetworks:
    policy_network: networks.FeedForwardNetwork
    qr_network: networks.FeedForwardNetwork
    model_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_world_model_ensemble(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (512, 512),
    activation: networks.ActivationFn = linen.swish,
    learn_std: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a model network."""

    # Convert obs_size to integer if it's a shape tuple
    if isinstance(obs_size, (tuple, list)):
        obs_size = obs_size[0] if len(obs_size) == 1 else sum(obs_size)
    obs_size = int(obs_size)

    class MModule(linen.Module):
        """M Module."""

        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            net = networks.MLP
            # FIXME (yarden): +2
            single_output_dim = obs_size + 1  # +2 for reward and cost
            if learn_std:
                single_output_dim *= 2
            output_dim = single_output_dim
            hidden_dims = list(hidden_layer_sizes) + [output_dim]
            out = net(
                layer_sizes=hidden_dims,
                activation=activation,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            return out

    model = MModule(obs_size=obs_size)

    def apply(preprocessor_params, params, obs, actions):
        obs_processed = preprocess_observations_fn(obs, preprocessor_params)
        raw_output = model.apply(params, obs_processed, actions)
        # Std devs also need to match the shape (B, E, feature_dim)
        if not learn_std:
            diff_obs_raw, reward = (
                raw_output[..., :obs_size],
                raw_output[..., obs_size],
            )
            obs = postprocess_observations_fn(
                diff_obs_raw + obs_processed, preprocessor_params
            )
            obs_std = jnp.ones_like(obs) * 1e-3
            reward_std = jnp.ones_like(reward) * 1e-3
            # FIXME (yarden): like cost
            cost_std = jnp.ones_like(reward) * 1e-3
        else:
            # FIXME (yarden): costs are not taken
            means, stds = jnp.split(raw_output, 2, axis=-1)
            diff_obs_raw, reward = means[..., :obs_size], means[..., obs_size]
            # FIXME (yarden)
            # obs = postprocess_observations_fn(
            #     diff_obs_raw + obs_processed, preprocessor_params
            # )
            obs = postprocess_observations_fn(
                diff_obs_raw + obs_processed, preprocessor_params
            )
            obs_std, reward_std = stds[..., :obs_size], stds[..., obs_size]
            # FIXME: (manu) figure out postprocessing of stds (only take scaling part of postprocessor)
            # FIXME (yarden): pass the costs not zeros like reward
        return (obs, reward, jnp.zeros_like(reward)), (obs_std, reward_std, cost_std)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    net = networks.FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), apply=apply
    )
    return net


def make_mbpo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    postprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    model_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    value_obs_key: str = "state",
    policy_obs_key: str = "state",
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
) -> MBPONetworks:
    """Make SAC networks."""
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
    qr_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    model_network = make_world_model_ensemble(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        postprocess_observations_fn=postprocess_observations_fn,
        hidden_layer_sizes=model_hidden_layer_sizes,
        activation=activation,
        # FIXME (yarden): don't hardcode this
        learn_std=False,
    )
    return MBPONetworks(
        policy_network=policy_network,
        qr_network=qr_network,
        model_network=model_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
