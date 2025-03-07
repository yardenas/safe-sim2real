import jax.numpy as jnp
from brax.envs import Wrapper
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


def make(**kwargs):
    env = locomotion.load(name, **kwargs)
    env = FlipConstraintWrapper(env)
    return env


locomotion.register_environment(
    f"Safe{name}", make, locomotion.go1_joystick.default_config
)
locomotion.ALL += [f"Safe{name}"]
