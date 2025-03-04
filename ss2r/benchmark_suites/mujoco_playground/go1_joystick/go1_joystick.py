import jax.numpy as jnp
from brax.envs import Wrapper
from mujoco_playground import locomotion


def domain_randomization(sys, rng, cfg):
    model, in_axes = locomotion.domain_randomization(sys, rng, cfg)
    new_pair_friction = model.pair_friction[:, :2, 0:2]
    new_dof_frictionloss = model.dof_frictionloss[6:]
    new_armature = model.dof_armature[6:]
    new_body_mass = model.body_mass
    new_qpos0 = model.qpos0[7:]
    samples = jnp.stack(
        [
            new_pair_friction,
            new_dof_frictionloss,
            new_armature,
            new_body_mass,
            new_qpos0,
        ],
    )
    return model, in_axes, samples


class SampleCommand(Wrapper):
    def __init__(self, env, frequency_factor=1):
        super().__init__(env)
        if not isinstance(frequency_factor, int) or frequency_factor < 1:
            raise ValueError("frequency_factor must be an integer greater than 0")
        self.frequency_factor = frequency_factor

    def step(self, state, action):
        state = self.env.step(state, action)
        current_step = state.info["steps_until_next_cmd"]
        # First undo -1 by the original step function
        correct_step = (current_step + 1) - self.frequency_factor
        state.info["steps_until_next_cmd"] = correct_step
        return state
