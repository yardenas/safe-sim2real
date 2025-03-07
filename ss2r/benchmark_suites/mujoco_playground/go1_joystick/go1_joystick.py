import jax
import jax.numpy as jnp
from brax.envs import Wrapper
from mujoco_playground import locomotion
from brax.envs.base import Wrapper, Env, State


def domain_randomization(sys, rng, cfg):

    FLOOR_GEOM_ID = 0
    TORSO_BODY_ID = 1
    model = sys

    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0). <- Original params
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=cfg.floor_friction[0], maxval=cfg.floor_friction[1])
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(12,), minval=cfg.scale_static_friction[0], maxval=cfg.scale_static_friction[1]
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(12,), minval=cfg.scale_armature[0], maxval=cfg.scale_armature[1]
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Jitter center of mass positiion: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=cfg.jitter_center_of_mass[0], maxval=cfg.jitter_center_of_mass[0])
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(model.nbody,), minval=cfg.scale_link_masses[0], maxval=cfg.scale_link_masses[1]
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-1.0, 1.0).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=cfg.add_torso_mass[0], maxval=cfg.add_torso_mass[1])
        body_mass = body_mass.at[TORSO_BODY_ID].set(
            body_mass[TORSO_BODY_ID] + dmass
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(key, shape=(12,), minval=cfg.jitter_qpos0[0], maxval=cfg.jitter_qpos0[1])
        )

        return (
            geom_friction,
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        )

    (
        friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "body_ipos": 0,
        "body_mass": 0,
        "qpos0": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
    })

    model = model.tree_replace({
        "geom_friction": friction,
        "body_ipos": body_ipos,
        "body_mass": body_mass,
        "qpos0": qpos0,
        "dof_frictionloss": dof_frictionloss,
        "dof_armature": dof_armature,
    })

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


class FlipConstraintWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Make sure that flipping is only communicated to the agent via the
        # cost.
        self.env._config.reward_config.scales["termination"] = 0.0

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        fall = self.env._get_termination(state.data)
        state.info["cost"] = fall.astype(jnp.float32)
        return state


name = "Go1JoystickFlatTerrain"


def make_joint(**kwargs):
    env = locomotion.load(name, **kwargs)
    env = JointConstraintWrapper(env)
    return env

def make_flip(**kwargs):
    env = locomotion.load(name, **kwargs)
    env = FlipConstraintWrapper(env)
    return env

locomotion.register_environment(
    f"SafeJoint{name}", make_joint, locomotion.go1_joystick.default_config
)
locomotion.register_environment(
    f"SafeFlip{name}", make_flip, locomotion.go1_joystick.default_config
)
locomotion.ALL += [f"SafeJoint{name}"]
locomotion.ALL += [f"SafeFlip{name}"]
