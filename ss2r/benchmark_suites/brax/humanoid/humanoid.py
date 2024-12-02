# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Trains a humanoid to run in the +x direction."""
import functools
import os

import jax
import jax.numpy as jnp
import mujoco
from brax import base
from brax.envs import Env, PipelineEnv, Wrapper, humanoid, register_environment
from brax.envs.base import State
from brax.io import mjcf


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        # Hardcoding _POLE_MASS to avoid weird jax issues.
        pole_mass = jnp.asarray([1.0, 0.1])
        mask = jnp.asarray([0.0, 1.0])
        mass_sample = (
            jax.random.uniform(rng, minval=cfg.mass[0], maxval=cfg.mass[1]) * mask
        )
        mass_sample = pole_mass + mass_sample
        rng, _ = jax.random.split(rng)
        gear = sys.actuator.gear.copy()[0]
        gear_sample = (
            jax.random.uniform(rng, minval=cfg.gear[0], maxval=cfg.gear[1]) + gear
        )
        return mass_sample, gear_sample

    mass_sample, actuator_gear = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "link.inertia.mass": 0,
            "actuator.gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "link.inertia.mass": mass_sample,
            "actuator.gear": actuator_gear[:, None],
        }
    )
    samples = jnp.stack(
        [
            mass_sample[:, -1],
            actuator_gear,
        ],
        axis=-1,
    )
    return sys, in_axes, samples


def domain_randomization_length(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

    pos = randomize(rng)
    sys_v = sys.tree_replace({"link.inertia.transform.pos": pos})
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"link.inertia.transform.pos": 0})
    return sys_v, in_axes, pos


class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env, max_force: float):
        assert isinstance(env, Humanoid)
        super().__init__(env)
        self.max_force = max_force

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        cost = nstate.done
        nstate.info["cost"] = cost
        return nstate


class Humanoid(humanoid.Humanoid):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        backend="mjx",
        **kwargs,
    ):
        dir = os.path.dirname(__file__)
        path = os.path.join(dir, "humanoid.xml")
        sys = mjcf.load(path)
        self.head_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SENSOR.value, "head_touch"
        )
        self.torso_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_SENSOR.value, "torso_touch"
        )

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.0015})
            n_frames = 10
            gear = jnp.array(
                [
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        PipelineEnv.__init__(self, sys=sys, backend=backend, **kwargs)
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def head_touch(self, pipeline_state: base.State) -> jax.Array:
        return jnp.linalg.norm(pipeline_state.sensordata[self.head_id])

    def torso_touch(self, pipeline_state: base.State) -> jax.Array:
        return jnp.linalg.norm(pipeline_state.sensordata[self.torso_id])


for safe in [True, False]:
    name = ["humanoid"]
    safe_str = "safe" if safe else ""

    def make(safe, **kwargs):
        max_force = kwargs.pop("max_force", 30.0)
        env = Humanoid(**kwargs)
        if safe:
            env = ConstraintWrapper(env, max_force)
        return env

    if safe:
        name.append("safe")
    name_str = "_".join(name)
    register_environment(name_str, functools.partial(make, safe))
