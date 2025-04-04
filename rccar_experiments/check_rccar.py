import logging
import pickle

import hydra
import imageio
import jax
import numpy as np

from rccar_experiments.utils import (
    DummyPolicy,
    Trajectory,
    collect_trajectory,
    make_env,
)
from ss2r.benchmark_suites.rccar.rccar import decode_angles, draw_scene

_LOG = logging.getLogger(__name__)


def load_trajectory(path):
    with open(path, "rb") as f:
        traj = pickle.load(f)
    return traj


def decode_angle(trajectory):
    obs = decode_angles(trajectory.observation, 2)
    return Trajectory(
        observation=obs,
        action=trajectory.action,
        reward=trajectory.reward,
        cost=trajectory.cost,
    )


def play_recorded_trajectory(actions, env):
    _, trajectory = collect_trajectory(env, DummyPolicy(actions), jax.random.PRNGKey(0))
    return trajectory


@hydra.main(version_base=None, config_path="../ss2r/configs", config_name="check_rccar")
def main(cfg):
    baseline_trajectory = load_trajectory(cfg.baseline_trajectory)
    check_trajectory = load_trajectory(cfg.check_trajectory)
    cfg.episode_length = len(baseline_trajectory.action)
    model = make_env(cfg)
    model.step = jax.jit(model.step)
    asarray = lambda x: np.asarray(x)
    baseline_trajectory = jax.tree_map(
        asarray, baseline_trajectory, is_leaf=lambda x: isinstance(x, list)
    )
    check_trajectory = jax.tree_map(
        asarray, check_trajectory, is_leaf=lambda x: isinstance(x, list)
    )
    model_trajectory = play_recorded_trajectory(check_trajectory.action, model)
    model_trajectory = jax.tree_map(
        asarray, model_trajectory, is_leaf=lambda x: isinstance(x, list)
    )
    if cfg.decode_check:
        check_trajectory = decode_angle(check_trajectory)
    model_trajectory = decode_angle(model_trajectory)
    horizon = min(
        len(baseline_trajectory.observation),
        len(check_trajectory.observation),
        len(model_trajectory.observation),
    )
    vid = []
    obstacle_position, obstacle_radius = model.obstacle[:2], model.obstacle[2]
    for i in range(horizon):
        baseline = draw_scene(
            baseline_trajectory.observation[:, :6],
            i,
            obstacle_position,
            obstacle_radius,
        )
        check = draw_scene(
            check_trajectory.observation[:, :6], i, obstacle_position, obstacle_radius
        )
        model = draw_scene(
            model_trajectory.observation[:, :6],
            i,
            obstacle_position,
            obstacle_radius,
        )
        vid.append((baseline, check, model))
    vid = np.asarray(vid)
    baseline, check, model = np.split(vid, 3, axis=1)
    vid = np.concatenate([baseline, check, model], axis=3).squeeze()
    imageio.mimsave("check_rccar.gif", vid, fps=30, loop=0)
    if not np.allclose(
        baseline_trajectory.observation[:horizon, :6],
        check_trajectory.observation[:horizon, :6],
        atol=0.1,
        rtol=0.05,
    ):
        _LOG.error("Baseline and check trajectories do not match")
    else:
        _LOG.info("Baseline and check trajectories match, congratulations!")


if __name__ == "__main__":
    main()
