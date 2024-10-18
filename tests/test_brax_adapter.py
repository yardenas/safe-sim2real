import pytest

import jax
import jax.numpy as jnp

from ss2r import benchmark_suites
from ss2r.benchmark_suites.brax import BraxAdapter
from tests import make_test_config

_ENVS = 128


@pytest.fixture
def adapter(request) -> BraxAdapter:
    if request.param is None:
        domain_randomization = True
    else:
        domain_randomization = request.param
    cfg = make_test_config(
        [
            f"training.parallel_envs={_ENVS}",
            f"environment.brax.domain_randomization={str(domain_randomization)}",
            "environment/task=inverted_pendulum",
        ]
    )
    make_env = benchmark_suites.make(cfg)
    dummy_env = make_env()
    assert isinstance(dummy_env, BraxAdapter)
    return dummy_env


@pytest.mark.parametrize(
    "adapter,similar_rate", [(True, 0.0), (False, 1.0)], indirect=["adapter"]
)
def test_parameterization(adapter: BraxAdapter, similar_rate):
    def policy(*_, **__):
        return jnp.zeros((adapter.action_size,)), None

    policy = jax.vmap(policy, in_axes=(0, None))
    state = adapter.reset([0] * adapter.parallel_envs)
    next_state, _ = adapter.step(state, policy, jax.random.PRNGKey(0))
    count = sum(
        jnp.allclose(next_state.obs[j, :], next_state.obs[i, :])
        for i in range(adapter.parallel_envs)
        for j in range(adapter.parallel_envs)
        if i != j
    )
    assert (
        count / _ENVS**2 <= similar_rate
    ), f"Domain randomization {adapter.parameterizations is not None}, expected {similar_rate}"


def test_set_state(adapter: BraxAdapter):
    state = adapter.reset([0] * adapter.parallel_envs).pipeline_state
    qp = jnp.concatenate([state.q, state.qd], -1)
    purturbed_qp = qp + jnp.ones_like(qp)
    next_state = adapter.set_state(purturbed_qp)
    new_purturbed_qp = jnp.concatenate(
        [next_state.pipeline_state.q, next_state.pipeline_state.qd], -1
    )
    assert jnp.allclose(purturbed_qp, new_purturbed_qp)
