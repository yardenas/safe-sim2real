import pickle

import hydra
import jax

from rccar_experiments.utils import collect_trajectory
from ss2r import benchmark_suites
from ss2r.rl.utils import DummyPolicy


def load_trajectory(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def play_recorded_trajectory(actions, env):
    trajectory = collect_trajectory(
        env, DummyPolicy(actions), DummyPolicy(actions), jax.random.PRNGKey(0)
    )
    return trajectory


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="check_rccar")
def main(cfg):
    baseline_trajectory = load_trajectory(cfg.baseline_trajectory)
    check_trajectory = load_trajectory(cfg.check_trajectory)
    if cfg.compare_with_model:
        train_env, _ = benchmark_suites.make(cfg)
        env_trajectory = play_recorded_trajectory(baseline_trajectory, train_env)


if __name__ == "__main__":
    main()
