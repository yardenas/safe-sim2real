from typing import Tuple

import jax
from brax import envs
from brax.training import gradients
from brax.training.types import (
    PRNGKey,
)

from ss2r.algorithms.mb_ppo import Metrics, TrainingState
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.sac.types import ReplayBufferState, float32


def update_fn(
    model_loss_fn,
    model_optimizer,
    replay_buffer,
    learn_std,
):
    model_update_fn = gradients.gradient_update_fn(
        model_loss_fn,
        model_optimizer,
        pmap_axis_name=None,
        has_aux=True,
    )

    def training_step(
        carry: Tuple[TrainingState, ReplayBufferState, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, buffer_state, training_key = carry
        key_sgd, new_key = jax.random.split(training_key, 2)
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = float32(transitions)
        (_, aux), model_params, model_optimizer_state = model_update_fn(
            training_state.params.model,
            training_state.normalizer_params,
            transitions,
            key_sgd,
            learn_std,
            optimizer_state=training_state.optimizer_state[0],  # type: ignore
        )
        new_training_state = TrainingState(
            optimizer_state=(model_optimizer_state,)
            + training_state.optimizer_state[1:],
            params=mb_ppo_losses.MBPPOParams(  # type: ignore
                model=model_params,
                policy=training_state.params.policy,
                value=training_state.params.value,
                cost_value=training_state.params.cost_value,
            ),
            normalizer_params=training_state.normalizer_params,
            env_steps=training_state.env_steps,
        )  # type: ignore
        return (new_training_state, buffer_state, new_key), aux

    return training_step
