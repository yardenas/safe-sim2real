import functools

import jax
import jax.numpy as jnp
import mujoco
from brax.envs import Env, Wrapper, register_environment, walker2d
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


deg_to_rad = lambda deg: deg * jnp.pi / 180


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
        rng = jax.random.split(rng, 3)
        # Ensure symmetry
        actuator_ids = get_actuators_by_joint_names(
            sys,
            [
                "thigh_joint",
                "leg_joint",
                "foot_joint",
                "thigh_left_joint",
                "leg_left_joint",
                "foot_left_joint",
            ],
        )
        gear_sample = sys.actuator.gear.copy()
        thigh = jax.random.uniform(
            rng[0], minval=deg_to_rad(cfg.thigh[0]), maxval=deg_to_rad(cfg.thigh[1])
        )
        leg = jax.random.uniform(
            rng[1], minval=deg_to_rad(cfg.leg[0]), maxval=deg_to_rad(cfg.leg[1])
        )
        foot = jax.random.uniform(
            rng[2], minval=deg_to_rad(cfg.foot[0]), maxval=deg_to_rad(cfg.foot[1])
        )
        name_values = {
            "thigh_joint": thigh,
            "leg_joint": leg,
            "foot_joint": foot,
            "thigh_left_joint": thigh,
            "leg_left_joint": leg,
            "foot_left_joint": foot,
        }
        for name, value in name_values.items():
            actuator_id = actuator_ids[name]
            gear_sample = gear_sample.at[actuator_id].add(value)
        return (
            friction_sample,
            gear_sample,
            jnp.stack(
                [friction, thigh, leg, foot],
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
        assert isinstance(env, walker2d.Walker2d)
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jnp.pi / 180
        joint_names = [
            "thigh_joint",
            "leg_joint",
            "foot_joint",
            "thigh_left_joint",
            "leg_left_joint",
            "foot_left_joint",
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
            normalized_angle = normalize_angle(angle)
            lower_limit = normalize_angle(joint_range[0] - self.angle_tolerance)
            upper_limit = normalize_angle(joint_range[1] + self.angle_tolerance)
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
    name = ["walker2d"]
    safe_str = "safe" if safe else ""

    def make(safe, **kwargs):
        angle_tolerance = kwargs.pop("angle_tolerance", 30.0)
        angle_tolerance = deg_to_rad(angle_tolerance)
        env = walker2d.Walker2d(**kwargs)
        if safe:
            env = ConstraintWrapper(env, angle_tolerance)
        return env

    if safe:
        name.append("safe")
    name_str = "_".join(name)
    register_environment(name_str, functools.partial(make, safe))
