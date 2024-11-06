import pickle

import hydra
import imageio
import jax
import numpy as np

from rccar_experiments.utils import DummyPolicy, collect_trajectory, make_env
from ss2r.benchmark_suites.rccar.rccar import draw_scene


def load_trajectory(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def play_recorded_trajectory(actions, env):
    _, trajectory = collect_trajectory(env, DummyPolicy(actions), jax.random.PRNGKey(0))
    return trajectory


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="check_rccar")
def main(cfg):
    baseline_trajectory = load_trajectory(cfg.baseline_trajectory)
    check_trajectory = load_trajectory(cfg.check_trajectory)
    model = make_env(cfg)
    model_trajectory = play_recorded_trajectory(baseline_trajectory.action, model)
    horizon = min(
        len(baseline_trajectory.observation),
        len(check_trajectory.observation),
        len(model_trajectory.observation),
    )
    vid = []
    obstacle_position, obstacle_radius = model.obstacle[:2], model.obstacle[2]
    for i in range(horizon):
        baseline = draw_scene(
            baseline_trajectory.observation[:, :7],
            i,
            obstacle_position,
            obstacle_radius,
        )
        check = draw_scene(
            check_trajectory.observation[:, :7], i, obstacle_position, obstacle_radius
        )
        model = draw_scene(
            np.asarray(model_trajectory.observation)[:, :7],
            i,
            obstacle_position,
            obstacle_radius,
        )
        vid.append((baseline, check, model))
    vid = np.asarray(vid)
    baseline, check, model = np.split(vid, 3, axis=1)
    vid = np.concatenate([baseline, check, model], axis=3).squeeze()
    imageio.mimsave("check_rccar.gif", vid, fps=30)


if __name__ == "__main__":
    main()
