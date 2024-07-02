from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree


class Learner:
    def __init__(
        self, model: PyTree, optimizer_config: dict[str, Any], batched: bool = False
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.get("clip", float("inf"))),
            optax.scale_by_adam(eps=optimizer_config.get("eps", 1e-8)),
            optax.scale(-optimizer_config.get("lr", 1e-3)),
        )
        if batched:
            init_fn = eqx.filter_vmap(lambda model: self.optimizer.init(model))
        else:
            init_fn = self.optimizer.init
        self.state = init_fn(eqx.filter(model, eqx.is_array))

    def grad_step(
        self, model: PyTree, grads: PyTree, state: optax.OptState
    ) -> tuple[PyTree, optax.OptState]:
        updates, new_opt_state = self.optimizer.update(grads, state)
        all_ok = all_finite(updates)
        updates = update_if(
            all_ok, updates, jax.tree_map(lambda x: jnp.zeros_like(x), updates)
        )
        new_opt_state = update_if(all_ok, new_opt_state, state)
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state


def all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(True)
    else:
        leaves = list(map(jnp.isfinite, leaves))
        leaves = list(map(jnp.all, leaves))
        return jnp.stack(list(leaves)).all()


def update_if(pred, update, fallback):
    return jax.tree_map(lambda x, y: jax.lax.select(pred, x, y), update, fallback)
