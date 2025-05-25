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

from typing import Sequence, Tuple, Callable

import jax
import jax.numpy as jnp
import flax
from flax import linen
from brax.training import distribution, networks, types
from brax.training.types import PRNGKey



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
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
            x = linen.LayerNorm()(x)
            x = self.activation(x)
            x = linen.Dense(features=self.hidden_dims[0], kernel_init=self.kernel_init)(x)
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
    predict_std: bool = False,
) -> networks.FeedForwardNetwork:
    """Creates a model network."""

    class MModule(linen.Module):
        """M Module."""
        n_ensemble: int
        obs_size: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            # Remove print statements for JIT compatibility
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            net = BroNet if use_bro else networks.MLP
            output_dim = obs_size * 2 if predict_std else obs_size
            hidden_dims = list(hidden_layer_sizes) + [output_dim]
            for _ in range(self.n_ensemble):
                m = net(
                    hidden_dims=hidden_dims,
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(m)
            ensemble_output = jnp.concatenate(res, axis=-1)
            return ensemble_output
        
    model = MModule(n_ensemble=n_ensemble, obs_size=obs_size)

    @jax.jit
    def apply(processor_params, params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        raw_output = model.apply(params, obs, actions)
        batch_size = obs.shape[0]
        if predict_std:
            reshaped = raw_output.reshape(batch_size, n_ensemble, obs_size * 2)
            pred_mean = reshaped[:, :, :obs_size]
            pred_std = reshaped[:, :, obs_size:]
            return pred_mean, pred_std
        else:
            pred = raw_output.reshape(batch_size, n_ensemble, obs_size)
            std = jnp.ones_like(pred) * 1e-3
            return pred, std

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: model.init(key, dummy_obs, dummy_action), 
        apply=apply
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
    )

    return MBPPONetworks(
        model_network=model_network,
        policy_network=policy_network,
        value_network=value_network,
        cost_value_network=cost_value_network,
        parametric_action_distribution=parametric_action_distribution,
    )  # type: ignore
