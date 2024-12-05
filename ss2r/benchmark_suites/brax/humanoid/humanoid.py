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
        rng, rng_ = jax.random.split(rng)
        friction = jax.random.uniform(
            rng_, minval=cfg.friction[0], maxval=cfg.friction[1]
        )
        friction_sample = sys.geom_friction.copy()
        friction_sample = friction_sample.at[0, 0].add(friction)
        friction_sample = jnp.clip(friction_sample, a_min=0.0, a_max=1.0)
        rng = jax.random.split(rng, 4)
        # Ensure symmetry
        names_ids = {
            k: mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, k)
            for k in [
                "right_hip_x",
                "left_hip_x",
                "right_hip_y",
                "left_hip_y",
                "right_hip_z",
                "left_hip_z",
                "left_knee",
                "right_knee",
            ]
        }
        gear_sample = sys.actuator.gear.copy()
        hip_x = jax.random.uniform(rng[0], minval=cfg.hip.x[0], maxval=cfg.hip.x[1])
        hip_y = jax.random.uniform(rng[1], minval=cfg.hip.y[0], maxval=cfg.hip.y[1])
        hip_z = jax.random.uniform(rng[2], minval=cfg.hip.z[0], maxval=cfg.hip.z[1])
        knee = jax.random.uniform(rng[3], minval=cfg.knee[0], maxval=cfg.knee[1])
        name_values = {
            "right_hip_x": hip_x,
            "left_hip_x": hip_x,
            "right_hip_y": hip_y,
            "left_hip_y": hip_y,
            "right_hip_z": hip_z,
            "left_hip_z": hip_z,
            "left_knee": knee,
            "right_knee": knee,
        }
        for name, value in name_values.items():
            actuator_id = names_ids[name]
            gear_sample = gear_sample.at[actuator_id].add(value)
        return (
            friction_sample,
            gear_sample,
            jnp.stack([friction, hip_x, hip_y, hip_z, knee]),
        )

    friction_sample, gear_sample, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator.gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_friction": friction_sample,
            "actuator.gear": gear_sample,
        }
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
    def __init__(self, env: Env, max_angle_scale: float):
        assert isinstance(env, Humanoid)
        super().__init__(env)
        self.max_angle_scale = max_angle_scale
        joint_names = [
            "abdomen_z",
            "abdomen_y",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ]
        self.joint_ids = jnp.asarray(
            [
                self.env.sys.mj_model.jnt_qposadr[
                    mujoco.mj_name2id(
                        env.sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name
                    )
                ]
                for name in joint_names
            ]
        )
        self.joint_ranges = self.env.sys.jnt_range[1:] * 180 / jnp.pi

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        joint_angles = nstate.pipeline_state.qpos[self.joint_ids]
        cost = jnp.zeros_like(nstate.reward)
        for angle, joint_range in zip(joint_angles, self.joint_ranges):
            cost += (
                (angle < (joint_range[0] * self.max_angle_scale))
                | (angle >= (joint_range[1] * self.max_angle_scale))
            ).astype(jnp.float32)
        nstate.info["cost"] = (cost > 0).astype(jnp.float32)
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
        max_angle_scale = kwargs.pop("max_angle_scale", 30.0)
        env = Humanoid(**kwargs)
        if safe:
            env = ConstraintWrapper(env, max_angle_scale)
        return env

    if safe:
        name.append("safe")
    name_str = "_".join(name)
    register_environment(name_str, functools.partial(make, safe))
