from hydra import compose, initialize
import numpy as np

from brax.training.types import Policy

from ss2r.rl.types import Report, Simulator


class DummyAgent:
    def __init__(self, action_size, config) -> None:
        self.config = config
        parallel_envs = config.training.parallel_envs
        self._policy = lambda *_: (np.repeat(np.zeros(action_size), parallel_envs), None)

    @property
    def policy(self) -> Policy:
        return self._policy

    def train(self, steps: int, simulator: Simulator) -> None:
        return Report({}, {})


def make_test_config(additional_overrides=None):
    if additional_overrides is None:
        additional_overrides = []
    with initialize(version_base=None, config_path="../ss2r/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "writers=[stderr]",
                "+experiment=debug",
            ]
            + additional_overrides,
        )
        return cfg
