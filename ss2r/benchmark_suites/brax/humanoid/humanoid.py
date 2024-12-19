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

import jax
import jax.numpy as jnp
import mujoco
from brax.envs import Env, Wrapper, humanoid, register_environment
from brax.envs.base import State


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


class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env, angle_tolerance: float):
        assert isinstance(env, humanoid.Humanoid)
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jnp.pi / 180
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
        self.joint_ranges = self.env.sys.jnt_range[1:]

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        with jax.disable_jit(False):
            nstate = jax.jit(self.env.step)(state, action)
        joint_angles = nstate.pipeline_state.qpos[self.joint_ids]
        cost = jnp.zeros_like(nstate.reward)
        for i, (angle, joint_range) in enumerate(zip(joint_angles, self.joint_ranges)):
            normalized_angle = normalize_angle(
                angle, lower_bound=-jnp.pi, upper_bound=jnp.pi
            )
            lower_limit = normalize_angle(
                joint_range[0] - self.angle_tolerance,
                lower_bound=-jnp.pi,
                upper_bound=jnp.pi,
            )
            upper_limit = normalize_angle(
                joint_range[1] + self.angle_tolerance,
                lower_bound=-jnp.pi,
                upper_bound=jnp.pi,
            )
            is_out_of_range_case1 = (normalized_angle < lower_limit) & (
                normalized_angle >= upper_limit
            )
            is_out_of_range_case2 = (normalized_angle < lower_limit) | (
                normalized_angle >= upper_limit
            )
            out_of_range = jnp.where(
                upper_limit < lower_limit, is_out_of_range_case1, is_out_of_range_case2
            )
            cost += out_of_range
        nstate.info["cost"] = (cost > 0).astype(jnp.float32)
        return nstate


def normalize_angle(angle, lower_bound=-jnp.pi, upper_bound=jnp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


for safe in [True, False]:
    name = ["humanoid"]
    safe_str = "safe" if safe else ""

    def make(safe, **kwargs):
        angle_tolerance = kwargs.pop("angle_tolerance", 30.0)
        env = humanoid.Humanoid(**kwargs)
        if safe:
            env = ConstraintWrapper(env, angle_tolerance)
        return env

    if safe:
        name.append("safe")
    name_str = "_".join(name)
    register_environment(name_str, functools.partial(make, safe))
