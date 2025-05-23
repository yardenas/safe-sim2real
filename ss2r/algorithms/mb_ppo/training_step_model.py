import functools
import jax
import jax.numpy as jnp
from brax.training import types, gradients
from brax.training.acme import running_statistics
from brax.training.types import PRNGKey
from ss2r.algorithms.ppo import _PMAP_AXIS_NAME
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses

def update_model(
        model_loss_fn, 
        model_optimizer,
        num_minibatches: int,
):
    
    model_update_fn = gradients.gradient_update_fn(
        model_loss_fn, model_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    
    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, key = carry
        (
            model_optimizer_state,
        ) = optimizer_state

        key, key_loss = jax.random.split(key)
        (_, aux), model_params, model_optimizer_state = model_update_fn(
            params.model,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=model_optimizer_state,
        )

        optimizer_state = (
            model_optimizer_state,
        )
        params = mb_ppo_losses.ModelParams(
            model_params
        )  # type: ignore
        return (optimizer_state, params, penalizer_params, key), aux
    

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, penalizer_params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, penalizer_params, _), aux = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, penalizer_params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, penalizer_params, key), aux