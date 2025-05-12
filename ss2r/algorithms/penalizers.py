from typing import Any, NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import optax

Params = TypeVar("Params")


class Penalizer(Protocol):
    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: Params,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        ...


class CRPOParams(NamedTuple):
    burnin: int


class CRPO:
    def __init__(self, eta: float) -> None:
        self.eta = eta

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: CRPOParams,
        *,
        rest: Any = None,
    ) -> tuple[jax.Array, dict[str, Any], CRPOParams]:
        active = jnp.greater(constraint + self.eta, 0.0) | jnp.greater(params.burnin, 0)
        if rest is not None:
            loss_constraint = rest
        else:
            loss_constraint = constraint
        actor_loss = jnp.where(
            active,
            actor_loss,
            -loss_constraint,
        )
        new_params = CRPOParams(jnp.clip(params.burnin - 1, a_min=-1))
        aux = {
            "crpo/burnin_counter": new_params.burnin,
            "crpo/active": active,
        }
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: Params
    ) -> tuple[dict[str, Any], Params]:
        return {}, params


class AugmentedLagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    penalty_multiplier: jax.Array


class AugmentedLagrangian:
    def __init__(self, penalty_multiplier_factor: float) -> None:
        self.penalty_multiplier_factor = penalty_multiplier_factor

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: AugmentedLagrangianParams,
    ) -> tuple[jax.Array, dict[str, Any], Params]:
        psi, cond = augmented_lagrangian(constraint, *params)
        new_params = update_augmented_lagrangian(
            cond, params.penalty_multiplier, self.penalty_multiplier_factor
        )
        aux = {
            "lagrangian_cond": cond,
            "lagrange_multiplier": new_params.lagrange_multiplier,
        }
        return actor_loss + psi, aux, new_params


def augmented_lagrangian(
    constraint: jax.Array,
    lagrange_multiplier: jax.Array,
    penalty_multiplier: jax.Array,
) -> jax.Array:
    # Nocedal-Wright 2006 Numerical Optimization, Eq. 17.65, p. 546
    # (with a slight change of notation)
    g = -constraint
    c = penalty_multiplier
    cond = lagrange_multiplier + c * g
    psi = jnp.where(
        jnp.greater(cond, 0.0),
        lagrange_multiplier * g + c / 2.0 * g**2,
        -1.0 / (2.0 * c) * lagrange_multiplier**2,
    )
    return psi, cond


def update_augmented_lagrangian(
    cond: jax.Array, penalty_multiplier: jax.Array, penalty_multiplier_factor: float
):
    new_penalty_multiplier = jnp.clip(
        penalty_multiplier * (1.0 + penalty_multiplier_factor), penalty_multiplier, 1.0
    )
    new_lagrange_multiplier = jnp.clip(cond, a_min=0.0, a_max=100.0)
    return AugmentedLagrangianParams(new_lagrange_multiplier, new_penalty_multiplier)


class LagrangianParams(NamedTuple):
    lagrange_multiplier: jax.Array
    optimizer_state: optax.OptState


class Lagrangian:
    def __init__(self, multiplier_lr: float) -> None:
        self.optimizer = optax.adam(learning_rate=multiplier_lr)
        self.learning_rate = multiplier_lr

    def __call__(
        self,
        actor_loss: jax.Array,
        constraint: jax.Array,
        params: LagrangianParams,
        *,
        rest: Any,
    ) -> tuple[jax.Array, dict[str, Any], LagrangianParams]:
        cost_advantage = -rest
        lagrange_multiplier = params.lagrange_multiplier
        actor_loss += lagrange_multiplier * cost_advantage
        aux: dict[str, Any] = {}
        new_params = params
        return actor_loss, aux, new_params

    def update(
        self, constraint: jax.Array, params: LagrangianParams
    ) -> tuple[jax.Array, LagrangianParams]:
        new_lagrange_multiplier = update_lagrange_multiplier(
            constraint, params.lagrange_multiplier, self.learning_rate
        )
        aux = {"lagrange_multiplier": new_lagrange_multiplier}
        return aux, LagrangianParams(new_lagrange_multiplier, params.optimizer_state)


def update_lagrange_multiplier(
    constraint: jax.Array, lagrange_multiplier: jax.Array, learning_rate: float
) -> jax.Array:
    new_multiplier = jnp.maximum(lagrange_multiplier - learning_rate * constraint, 0.0)
    return new_multiplier
