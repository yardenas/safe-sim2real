import pytest

import jax
import jax.numpy as jnp

from ss2r import benchmark_suites
from ss2r.benchmark_suites.brax import BraxAdapter
from tests import make_test_config

_ENVS = 128


@pytest.fixture
def adapter() -> BraxAdapter:
    cfg = make_test_config(
        [
            f"training.parallel_envs={_ENVS}",
            "environment.brax.domain_randomization=True",
            "environment/task=inverted_pendulum",
            # "environment.task.task_name=cartpole_balance",
        ]
    )
    make_env = benchmark_suites.make(cfg)
    dummy_env = make_env()
    assert isinstance(dummy_env, BraxAdapter)
    return dummy_env


def test_parameterization(adapter: BraxAdapter):
    def policy(*_, **__):
        return jnp.zeros((adapter.action_size,)), None

    policy = jax.vmap(policy, in_axes=(0, None))
    state = adapter.reset([0] * adapter.parallel_envs)
    next_state, _ = adapter.step(state, policy, jax.random.PRNGKey(0))
    count = sum(
        jnp.allclose(next_state.obs[0, :], obs[0, :])
        for obs in jnp.split(next_state.obs[1:, :], adapter.parallel_envs - 1, axis=0)
    )
    assert (
        count / _ENVS < 0.2
    ), "Different environment initializations should have different trajectories"


def test_set_state(adapter: BraxAdapter):
    state = adapter.reset([0] * adapter.parallel_envs).pipeline_state
    qp = jnp.concatenate([state.q, state.qd], -1)
    purturbed_qp = qp + jnp.ones_like(qp)
    next_state = adapter.set_state(purturbed_qp)
    new_purturbed_qp = jnp.concatenate(
        [next_state.pipeline_state.q, next_state.pipeline_state.qd], -1
    )
    assert jnp.allclose(purturbed_qp, new_purturbed_qp)
