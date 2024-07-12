import jax


def domain_randomization(sys, rng):
    @jax.vmap
    def randomize(rng):
        cpole = jax.random.normal(rng) * 0.05 + sys.link.inertia.mass[-1]
        mass = sys.link.inertia.mass.at[-1].set(cpole)
        return mass, cpole

    mass, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"link.inertia.mass": 0})
    sys = sys.tree_replace({"link.inertia.mass": mass})
    return sys, in_axes, samples[:, None]


def uniform_state_sampler():
    pass
