import jax
import jax.numpy as jnp
from mujoco_playground import locomotion
from brax.envs.base import Wrapper, Env, State


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

class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self._env = env

    def reset(self, rng: jax.Array) -> State:
        state = self._env.reset(rng)
        joint_cost = self._cost_joint_pos_limits(state.data.qpos[7:])
        state.info["cost"] = joint_cost
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = self._env.step(state, action)
        joint_cost = self._cost_joint_pos_limits(state.data.qpos[7:])
        state = state.replace(info={**state.info, 'cost': joint_cost})
        return state

def make(**kwargs):
    env = locomotion.load("Go1JoystickFlatTerrain", **kwargs)
    env = ConstraintWrapper(env)
    return env

locomotion.register_environment("SafeGo1JoystickFlatTerrain", make, locomotion.go1_joystick.default_config)
locomotion.ALL += ["SafeGo1JoystickFlatTerrain"]