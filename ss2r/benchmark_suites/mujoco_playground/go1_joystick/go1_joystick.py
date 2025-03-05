import jax.numpy as jnp
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
