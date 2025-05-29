import functools
from typing import Any, Callable, Optional, Tuple, TypeAlias

import flax
import jax.numpy as jnp
import optax
from brax import envs
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.types import Params

from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.sac.data import get_collection_fn


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: tuple[
        optax.OptState, optax.OptState, optax.OptState, Optional[optax.OptState]
    ]
    params: mb_ppo_losses.MBPPOParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


Metrics: TypeAlias = types.Metrics


TrainingStep: TypeAlias = Callable[
    [
        Tuple[TrainingState, envs.State, types.PRNGKey, int],
    ],
    Tuple[Tuple[TrainingState, envs.State, types.PRNGKey], Metrics],
]

TrainingStepFactory: Any

InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]


def get_train_fn(cfg, checkpoint_path, restore_checkpoint_path):
    import jax.nn as jnn

    from ss2r.algorithms.mb_ppo import networks as mb_ppo_networks
    from ss2r.algorithms.mb_ppo import train as mb_ppo

    agent_cfg = dict(cfg.agent)
    training_cfg = {
        k: v
        for k, v in cfg.training.items()
        if k
        not in [
            "render_episodes",
            "train_domain_randomization",
            "eval_domain_randomization",
            "render",
            "store_checkpoint",
            "wandb_id",  # Add this to the exclusion list
        ]
    }
    model_hidden_layer_sizes = agent_cfg.pop("model_hidden_layer_sizes")
    policy_hidden_layer_sizes = agent_cfg.pop("policy_hidden_layer_sizes")
    value_hidden_layer_sizes = agent_cfg.pop("value_hidden_layer_sizes")
    activation = getattr(jnn, agent_cfg.pop("activation"))
    learn_std = agent_cfg.pop("learn_std", False)
    value_obs_key = "privileged_state" if cfg.training.value_privileged else "state"
    policy_obs_key = "privileged_state" if cfg.training.policy_privileged else "state"
    del training_cfg["value_privileged"]
    del training_cfg["policy_privileged"]
    del agent_cfg["name"]
    if "data_collection" in agent_cfg:
        del agent_cfg["data_collection"]
    network_factory = functools.partial(
        mb_ppo_networks.make_mb_ppo_networks,
        model_hidden_layer_sizes=model_hidden_layer_sizes,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        learn_std=learn_std,
        value_obs_key=value_obs_key,
        policy_obs_key=policy_obs_key,
    )
    data_collection = get_collection_fn(cfg)
    train_fn = functools.partial(
        mb_ppo.train,
        **agent_cfg,
        **training_cfg,
        network_factory=network_factory,
        restore_checkpoint_path=restore_checkpoint_path,
        checkpoint_logdir=checkpoint_path,
        get_experience_fn=data_collection,
    )

    return train_fn
