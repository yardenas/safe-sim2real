from datetime import datetime
import functools
import pytest
import jax
from brax.io import image
from brax import envs
import brax.training.agents.sac.train as sac
import matplotlib.pyplot as plt

from ss2r import benchmark_suites
from ss2r.benchmark_suites.brax import BraxAdapter
from tests import make_test_config


@pytest.fixture
def adapter() -> BraxAdapter:
    cfg = make_test_config([f"training.parallel_envs={1}"])
    make_env = benchmark_suites.make(cfg)
    dummy_env = make_env()
    assert isinstance(dummy_env, BraxAdapter)
    return dummy_env


def pytrees_unstack(pytree):
    leaves, treedef = jax.tree_flatten(pytree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def test_rollout(adapter: BraxAdapter):
    def policy(_, key):
        return jax.random.uniform(
            key, shape=(adapter.action_size,), minval=-1.0, maxval=1.0
        ), None

    policy = jax.vmap(policy, in_axes=(0, None))
    _, data = adapter.rollout(policy, 100, 666, with_pipeline_state=True)
    trajectory = jax.tree_map(lambda x: x[:, 0], data.extras["pipeline_state"])  # type: ignore
    trajectory = pytrees_unstack(trajectory)
    video = image.render_array(adapter.environment.sys, trajectory)
    display_video(video)


def display_video(video, fps=30):
    delay = 1 / fps
    _, ax = plt.subplots()
    plt.ion()
    for frame in video:
        ax.imshow(frame)
        ax.axis("off")
        plt.pause(delay)
        plt.draw()
    plt.ioff()

@pytest.mark.skip
def test_sac():
    train_fn = functools.partial(
        sac.train,
        num_timesteps=10000000,
        num_evals=20,
        reward_scaling=30,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        discounting=0.997,
        learning_rate=6e-4,
        num_envs=2048,
        batch_size=512,
        grad_updates_per_step=32,
        max_replay_size=1048576,
        min_replay_size=8192,
        seed=1,
    )
    xdata, ydata = [], []
    times = [datetime.now()]
    plt.ion()
    plt.show()

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])
        plt.cla()
        plt.xlim([0, train_fn.keywords["num_timesteps"]])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        plt.draw()
        plt.pause(0.5)

    with jax.disable_jit(False):
        env = envs.get_environment(env_name="cartpole", sparse=False, swingup=True)
        make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
