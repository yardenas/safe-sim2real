import jax.numpy as jnp
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
