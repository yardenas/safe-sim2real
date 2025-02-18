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
from brax.envs import Env, Wrapper, ant, register_environment
from brax.envs.base import State


def get_actuators_by_joint_names(sys, joint_names):
    """
    Given a MuJoCo system and a list of joint names,
    returns a dictionary mapping joint names to actuator indices.
    """
    joint_to_actuator = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id == -1:
            print(f"Warning: Joint '{joint_name}' not found in the model.")
            continue
        # Find actuator(s) controlling this joint
        for actuator_id in range(len(sys.mj_model.actuator_trnid)):
            if sys.mj_model.actuator_trnid[actuator_id, 0] == joint_id:
                joint_to_actuator[joint_name] = actuator_id
    return joint_to_actuator


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
        actuator_ids = get_actuators_by_joint_names(
            sys,
            [
                "hip_1",
                "ankle_1",
                "hip_2",
                "ankle_2",
                "hip_3",
                "ankle_3",
                "hip_4",
                "ankle_4",
            ],
        )
        gear_sample = sys.actuator.gear.copy()
        hip_forward = jax.random.uniform(rng[0], minval=cfg.hip[0], maxval=cfg.hip[1])
        ankle_forward = jax.random.uniform(
            rng[1], minval=cfg.ankle[0], maxval=cfg.ankle[1]
        )
        hip_backward = jax.random.uniform(rng[2], minval=cfg.hip[0], maxval=cfg.hip[1])
        ankle_backward = jax.random.uniform(
            rng[3], minval=cfg.ankle[0], maxval=cfg.ankle[1]
        )
        name_values = {
            "hip_1": hip_forward,
            "ankle_1": ankle_forward,
            "hip_2": hip_forward,
            "ankle_2": ankle_forward,
            "hip_3": hip_backward,
            "ankle_3": ankle_backward,
            "hip_4": hip_backward,
            "ankle_4": ankle_backward,
        }
        for name, value in name_values.items():
            actuator_id = actuator_ids[name]
            gear_sample = gear_sample.at[actuator_id].add(value)
        return (
            friction_sample,
            gear_sample,
            jnp.stack(
                [friction, hip_forward, ankle_forward, hip_backward, ankle_backward]
            ),
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
        assert isinstance(env, ant.Ant)
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jnp.pi / 180
        joint_names = [
            "hip_1",
            "ankle_1",
            "hip_2",
            "ankle_2",
            "hip_3",
            "ankle_3",
            "hip_4",
            "ankle_4",
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
        nstate = self.env.step(state, action)
        joint_angles = nstate.pipeline_state.qpos[self.joint_ids]
        cost = jnp.zeros_like(nstate.reward)
        for _, (angle, joint_range) in enumerate(zip(joint_angles, self.joint_ranges)):
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
    name = ["ant"]
    safe_str = "safe" if safe else ""

    def make(safe, **kwargs):
        angle_tolerance = kwargs.pop("angle_tolerance", 30.0)
        env = ant.Ant(**kwargs)
        if safe:
            env = ConstraintWrapper(env, angle_tolerance)
        return env

    if safe:
        name.append("safe")
    name_str = "_".join(name)
    register_environment(name_str, functools.partial(make, safe))
