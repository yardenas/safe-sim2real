from typing import Callable, Optional

import optax
from brax.training import gradients


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
    )

    def f(*args, optimizer_state, params=None):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(
            grads, optimizer_state, params
        )
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f
