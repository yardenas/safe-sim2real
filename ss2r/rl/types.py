from typing import Protocol, Sequence, Tuple

from brax import envs
from brax.training.types import Policy, PolicyParams, PRNGKey, Transition


class MakePolicyFn(Protocol):
    def __call__(self, policy_params: PolicyParams) -> Policy:
        ...


class UnrollFn(Protocol):
    def __call__(
        self,
        env: envs.Env,
        env_state: envs.State,
        make_policy_fn: MakePolicyFn,
        policy_params: PolicyParams,
        key: PRNGKey,
        *,
        extra_fields: Sequence[str],
    ) -> Tuple[envs.State, Transition]:
        ...
