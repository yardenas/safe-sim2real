import pytest
import jax
from brax.io import image
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
    plt.show()
