import jax
import jax.numpy as jnp

_TORSO_ID = 1
_LEFT_THIGH_ID = 5
_RIGHT_THIGH_ID = 2


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        #  https://github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/environments/walker.py#L593
        pos = sys.body_pos.copy()
        torso_length_sample = jax.random.uniform(
            rng, minval=cfg.torso_length[0], maxval=cfg.torso_length[1]
        )
        torso_length_sample = jnp.clip(torso_length_sample, a_min=-0.3, a_max=0.4)
        length = 1.3 + torso_length_sample
        scale_factor = length / 1.3
        geom = sys.geom_size.at[_TORSO_ID, 1].add(torso_length_sample)
        pos = pos.at[_TORSO_ID, -1].add(torso_length_sample)
        mass = sys.body_mass.at[_TORSO_ID].multiply(scale_factor)
        inertia = sys.body_inertia.at[_TORSO_ID].multiply(scale_factor**3)
        pos = pos.at[_LEFT_THIGH_ID, -1].add(torso_length_sample)
        pos = pos.at[_RIGHT_THIGH_ID, -1].add(torso_length_sample)
        friction = jax.random.uniform(
            rng, minval=cfg.friction[0], maxval=cfg.friction[1]
        )
        friction = sys.geom_friction.at[:, 0].add(friction)
        return (
            pos,
            mass,
            inertia,
            geom,
            friction,
            scale_factor[None],
        )

    pos, mass, inertia, geom, friction, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_pos": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "geom_size": 0,
            "geom_friction": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_pos": pos,
            "body_mass": mass,
            "body_inertia": inertia,
            "geom_size": geom,
            "geom_friction": friction,
        }
    )
    return sys, in_axes
