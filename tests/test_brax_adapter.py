import pytest

import jax
import jax.numpy as jnp

from ss2r import benchmark_suites
from ss2r.benchmark_suites.brax import BraxAdapter
from tests import make_test_config


@pytest.fixture
def adapter() -> BraxAdapter:
    cfg = make_test_config()
    make_env = benchmark_suites.make(cfg)
    dummy_env = make_env()
    return dummy_env


def test_parameterization(adapter: BraxAdapter):
    def policy(*_, **__):
        return jnp.zeros((adapter.action_size,)), None

    policy = jax.vmap(policy, in_axes=(0, None))
    state = adapter.reset([0] * adapter.parallel_envs)
    next_state, _ = adapter.step(state, policy, jax.random.PRNGKey(0))
    assert all(
        not jnp.allclose(next_state.obs[0, :], obs[0, :])
        for obs in jnp.split(next_state.obs[1:, :], adapter.parallel_envs - 1, axis=0)
    ), "Different environment initializations should have different trajectories"
