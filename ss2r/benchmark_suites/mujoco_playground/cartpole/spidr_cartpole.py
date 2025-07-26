from typing import Any, Dict, Optional, Union

import jax.numpy as jnp
from ml_collections import config_dict
from mujoco_playground._src.dm_control_suite import cartpole

from ss2r.benchmark_suites.wrappers import SPiDR


class VisionSPiDRCartpole(cartpole.Balance):
    def __init__(
        self,
        randomization_fn: Any,
        config: config_dict.ConfigDict = cartpole.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(True, False, config, config_overrides)
        config["vision"] = False
        base_cartpole = cartpole.Balance(True, False, config, config_overrides)
        self._spidr_env = SPiDR(
            base_cartpole,
            randomization_fn,
            8,
            5e4,
            0.0,
        )

    def reset(self, rng):
        state = super().reset(rng)
        state.info["disagreement"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state, action):
        spidr_state = state.replace(
            obs=self._spidr_env.env._get_obs(state.data, state.info)
        )
        spidr_state = self._spidr_env.step(spidr_state, action)
        out_state = super().step(state, action)
        out_state.info["disagreement"] = spidr_state.info["disagreement"]
        return out_state
