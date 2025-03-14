import jax
import jax.numpy as jnp
from brax.envs import Env, State, Wrapper
from mujoco_playground import locomotion


def domain_randomization(sys, rng, cfg):
    model, in_axes = locomotion.go1_randomize.domain_randomize(sys, rng)
    geom_friction = model.geom_friction[:, 0, 0]
    body_ipos = model.body_ipos[:, 1]
    body_mass = model.body_mass
    qpos0 = model.qpos0[:, 7:]
    dof_frictionloss = model.dof_frictionloss[:, 6:]
    dof_armature = model.dof_armature[:, 6:]
    samples = jnp.hstack(
        [
            geom_friction[:, None],
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        ],
    )
    return model, in_axes, samples


class JointConstraintWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.env._config.reward_config.scales["dof_pos_limits"] = 0.0

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        joint_cost = self._cost_joint_pos_limits(state.data.qpos[7:])
        state.info["cost"] = joint_cost
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        joint_cost = self._cost_joint_pos_limits(state.data.qpos[7:])
        state = state.replace(info={**state.info, "cost": joint_cost})
        return state


class JointTorqueConstraintWrapper(Wrapper):
    def __init__(self, env: Env, limit: float):
        super().__init__(env)
        self.env._config.reward_config.scales["torques"] = 0.0
        self.limit = limit

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self.env.step(state, action)
        torques = state.data.actuator_force
        state.info["cost"] = jnp.clip((jnp.abs(torques) - self.limit), a_min=0.0).max()
        return state


class FlipConstraintWrapper(Wrapper):
    def __init__(self, env: Env, limit: float):
        super().__init__(env)
        self.env._config.reward_config.scales["orientation"] = 0.0
        self.limit = limit

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        xy = self.env.get_upvector(state.data)[:2]
        cost = jnp.sum(jnp.square(xy))
        state.info["cost"] = cost
        return state


name = "Go1JoystickFlatTerrain"


def make_joint(**kwargs):
    env = locomotion.load(name, **kwargs)
    env = JointConstraintWrapper(env)
    return env


def make_joint_torque(**kwargs):
    limit = kwargs["config"]["torque_limit"]
    env = locomotion.load(name, **kwargs)
    env = JointTorqueConstraintWrapper(env, limit)
    return env


def make_flip(**kwargs):
    limit = kwargs["config"]["roll_limit"]
    env = locomotion.load(name, **kwargs)
    env = FlipConstraintWrapper(env, limit)
    return env


locomotion.register_environment(
    f"SafeJoint{name}", make_joint, locomotion.go1_joystick.default_config
)
locomotion.register_environment(
    f"SafeJointTorque{name}", make_joint_torque, locomotion.go1_joystick.default_config
)
locomotion.register_environment(
    f"SafeFlip{name}", make_flip, locomotion.go1_joystick.default_config
)
