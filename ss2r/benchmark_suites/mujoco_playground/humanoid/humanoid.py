import jax
import jax.numpy as jnp
import mujoco
from brax.envs import Wrapper
from mujoco_playground import MjxEnv, State, dm_control_suite

_name_to_id = {
    "right_hip_x": 3,
    "left_hip_x": 9,
    "right_hip_y": 5,
    "left_hip_y": 11,
    "right_hip_z": 4,
    "left_hip_z": 10,
    "left_knee": 12,
    "right_knee": 6,
}


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
        rng = jax.random.split(rng, 8)
        # Ensure symmetry
        gear_sample = sys.actuator_gear.copy()
        gear_hip_x = jax.random.uniform(
            rng[4], minval=cfg.gear_hip.x[0], maxval=cfg.gear_hip.x[1]
        )
        gear_hip_y = jax.random.uniform(
            rng[5], minval=cfg.gear_hip.y[0], maxval=cfg.gear_hip.y[1]
        )
        gear_hip_z = jax.random.uniform(
            rng[6], minval=cfg.gear_hip.z[0], maxval=cfg.gear_hip.z[1]
        )
        gear_knee = jax.random.uniform(
            rng[7], minval=cfg.gear_knee[0], maxval=cfg.gear_knee[1]
        )
        name_values = {
            "right_hip_x": gear_hip_x,
            "left_hip_x": gear_hip_x,
            "right_hip_y": gear_hip_y,
            "left_hip_y": gear_hip_y,
            "right_hip_z": gear_hip_z,
            "left_hip_z": gear_hip_z,
            "left_knee": gear_knee,
            "right_knee": gear_knee,
        }
        for name, gear in name_values.items():
            actuator_id = _name_to_id[name]
            gear_sample = gear_sample.at[actuator_id].add(gear)
        return (
            friction_sample,
            gear_sample,
            jnp.stack(
                [
                    friction,
                    gear_hip_x,
                    gear_hip_y,
                    gear_hip_z,
                    gear_knee,
                ],
                axis=-1,
            ),
        )

    friction_sample, gear_sample, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "geom_friction": friction_sample,
            "actuator_gear": gear_sample,
        }
    )
    return sys, in_axes, samples


def normalize_angle(angle, lower_bound=-jnp.pi, upper_bound=jnp.pi):
    """Normalize angle to be within [lower_bound, upper_bound)."""
    range_width = upper_bound - lower_bound
    return (angle - lower_bound) % range_width + lower_bound


class ConstraintWrapper(Wrapper):
    def __init__(self, env: MjxEnv, angle_tolerance: float):
        super().__init__(env)
        self.angle_tolerance = angle_tolerance * jnp.pi / 180.0
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
                mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)
                for name in joint_names
            ]
        )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        cost = jnp.zeros_like(nstate.reward)
        for id in zip(self.joint_ids):
            qpos_id = self.env.mj_model.jnt_qposadr[id]
            joint_range = self.env.mj_model.jnt_range[id]
            angle = nstate.data.qpos[qpos_id]
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


def make_safe(**kwargs):
    limit = kwargs["config"]["angle_tolerance"]
    env = dm_control_suite.load("HumanoidWalk", **kwargs)
    env = ConstraintWrapper(env, limit)
    return env


dm_control_suite.register_environment(
    "SafeHumanoidWalk", make_safe, dm_control_suite.humanoid.default_config
)
