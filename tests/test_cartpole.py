import functools

import brax.training.agents.sac.train as sac
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax import envs
from dm_control import suite
from dm_control.rl.control import flatten_observation

from ss2r import benchmark_suites
from tests import make_test_config


@pytest.mark.skip
def test_sac():
    train_fn = functools.partial(
        sac.train,
        num_timesteps=400000,
        num_evals=4,
        episode_length=1000,
        num_envs=512,
        batch_size=256,
        grad_updates_per_step=512,
        seed=1,
    )
    xdata, ydata = [], []

    def progress(num_steps, metrics):
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        print(f"{num_steps}: {metrics['eval/episode_reward']}")

    env = envs.get_environment(env_name="cartpole_swingup")
    train_fn(environment=env, progress_fn=progress)
    assert ydata[-1] > 750


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


def set_dmc_initial_state(env, init_state):
    physics = env.physics
    with physics.reset_context():
        # Set cart and pole initial state [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        physics.named.data.qpos["slider"] = init_state[0]  # Cart position
        physics.named.data.qvel["slider"] = init_state[1]  # Cart velocity
        physics.named.data.qpos["hinge_1"] = init_state[2]  # Pole angle
        physics.named.data.qvel["hinge_1"] = init_state[3]  # Pole angular velocity
    env._task.after_step(physics)
    out_state = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=env._task.get_observation(physics),
    )
    return out_state


def test_brax_dmc_cartpole():
    jax.config.update("jax_enable_x64", True)
    num_steps = 1000
    brax_env = envs.get_environment(
        env_name="cartpole", backend="generalized", swingup=True
    )
    dmc_env = suite.load(domain_name="cartpole", task_name="swingup")
    brax_env.reset(rng=jax.random.PRNGKey(0))
    dmc_env.reset()
    # Define a custom initial state [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    initial_state = np.array([0.0, 0.0, np.pi, 0.0])
    # Set the initial state in both environments
    brax_state = set_brax_initial_state(brax_env, initial_state)
    dmc_state = set_dmc_initial_state(dmc_env, initial_state)
    brax_step = jax.jit(brax_env.step)
    for _ in range(num_steps):
        # Extract observations from Brax and DMC
        brax_obs = brax_state.obs
        dmc_obs = flatten_observation(dmc_state.observation)["observations"]
        # Compare state shapes
        assert brax_obs.shape == dmc_obs.shape
        # Compare states elementwise
        assert np.allclose(brax_obs, dmc_obs, atol=1e-2)
        # Get action from the shared policy
        action = shared_policy(brax_obs)
        # Step both environments with the same action
        brax_state = brax_step(brax_state, action)
        dmc_state = dmc_env.step(action)


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
            "environment=cartpole",
            "training.train_domain_randomization=" + str(use_domain_randomization),
            "environment.task_name=cartpole",
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
