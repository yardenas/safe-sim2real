import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax import envs

from ss2r import benchmark_suites
from tests import make_test_config


def shared_policy(state):
    pos = state[0]
    vel = state[1]
    action = -1.0 * pos - 0.1 * vel
    return np.array([action])


def set_brax_initial_state(env, init_state):
    q = jax.numpy.asarray([init_state[0], init_state[2]])
    qd = jax.numpy.asarray([init_state[1], init_state[3]])
    pipeline_state = env.pipeline_init(q, qd)
    obs = env._get_obs(pipeline_state)
    reward, done, cost, steps, truncation = jax.numpy.zeros(5)
    info = {
        "cost": cost,
        "steps": steps,
        "truncation": truncation,
        "first_pipeline_state": pipeline_state,
        "first_obs": obs,
    }
    return envs.State(pipeline_state, obs, reward, done, {}, info)


_ENVS = 256


@pytest.mark.parametrize(
    "use_domain_randomization,similar_rate",
    [(True, 0.1), (False, 1.0)],
    ids=["domain_randomization", "no_domain_randomization"],
)
def test_parameterization(use_domain_randomization, similar_rate):
    cfg = make_test_config(
        [
            f"training.num_envs={_ENVS}",
            "environment=rccar",
            "training.train_domain_randomization=" + str(use_domain_randomization),
        ]
    )
    env, *_ = benchmark_suites.make(cfg)
    keys = jnp.stack([jax.random.PRNGKey(0) for _ in range(_ENVS)])
    brax_state = env.reset(rng=keys)
    brax_step = jax.jit(env.step)
    brax_state = brax_step(brax_state, jnp.ones((_ENVS, env.action_size)))
    count = sum(
        jnp.allclose(brax_state.obs[j, :], brax_state.obs[i, :])
        for i in range(_ENVS)
        for j in range(_ENVS)
        if i != j
    )
    total = _ENVS * (_ENVS - 1)
    assert (
        (count / total <= similar_rate and use_domain_randomization)
        or (count / total >= similar_rate and not use_domain_randomization)
    ), f"Domain randomization {use_domain_randomization is not None}, expected {similar_rate}"
