import pytest

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
    policy = lambda *_: jnp.zeros((adapter.action_size,)), None
    adapter.rollout(policy, 10)
