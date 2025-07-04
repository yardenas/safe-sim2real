from typing import Mapping, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
from brax.training import distribution, networks, types
from flax import linen

from ss2r.algorithms.sac.networks import (
    ActivationFn,
    BroNet,
    Initializer,
    SafeSACNetworks,
)

_HIDDEN_DIM = 50


def make_q_vision_network(
    vision_ecoder: networks.VisionMLP,
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.swish,
    n_critics: int = 2,
    state_obs_key: str = "",
    use_bro: bool = True,
    n_heads: int = 1,
    head_size: int = 1,
):
    class QModule(linen.Module):
        encoder: networks.VisionMLP
        n_critics: int

        @linen.compact
        def __call__(self, obs, actions):
            hidden = self.encoder(obs)
            hidden = activation(hidden)
            hidden = linen.Dense(_HIDDEN_DIM)(hidden)
            hidden = linen.LayerNorm()(hidden)
            hidden = jnp.concatenate([hidden, actions], axis=-1)
            res = []
            net = BroNet if use_bro else networks.MLP
            for _ in range(self.n_critics):
                q = net(  # type: ignore
                    layer_sizes=list(hidden_layer_sizes) + [head_size],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                    # FIXME
                    # num_heads=n_heads,
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(encoder=vision_ecoder, n_critics=n_critics)

    def apply(processor_params, params, obs, actions):
        if state_obs_key:
            state_obs = preprocess_observations_fn(
                obs[state_obs_key],
                networks.normalizer_select(processor_params, state_obs_key),
            )
            obs = {**obs, state_obs_key: state_obs}
        return q_module.apply(params, obs, actions)

    dummy_obs = {
        key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
    }
    dummy_action = jnp.zeros((1, action_size))
    return networks.FeedForwardNetwork(
        init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply
    )


def make_sac_vision_networks(
    observation_size: Mapping[str, Tuple[int, ...]],
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    activation: ActivationFn = linen.swish,
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform(),
    layer_norm: bool = False,
    state_obs_key: str = "",
    normalise_channels: bool = False,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bro: bool = True,
    n_critics: int = 2,
    n_heads: int = 1,
    *,
    safe: bool = False,
) -> SafeSACNetworks:
    """Make SAC networks."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = networks.make_policy_network_vision(
        observation_size=observation_size,
        output_size=parametric_action_distribution.param_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        state_obs_key=state_obs_key,
        layer_norm=layer_norm,
    )
    critic_encoder = networks.VisionMLP(
        layer_sizes=[_HIDDEN_DIM],  # NatureCNN followed by a hidden of 50
        activation=activation,
        kernel_init=kernel_init,
        # FIXME
        # normalise_channels=normalise_channels,
        state_obs_key=state_obs_key,
        layer_norm=layer_norm,
        activate_final=False,
        policy_head=True,
    )
    qr_network = make_q_vision_network(
        vision_ecoder=critic_encoder,
        observation_size=observation_size,
        action_size=action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    if safe:
        qc_network = make_q_vision_network(
            vision_ecoder=critic_encoder,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            hidden_layer_sizes=value_hidden_layer_sizes,
            activation=activation,
            use_bro=use_bro,
            n_critics=n_critics,
            n_heads=n_heads,
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
