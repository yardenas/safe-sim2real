import jax


def domain_randomization(sys, rng):
    @jax.vmap
    def randomize(rng):
        cpole = jax.random.normal(rng) * 0.05 + sys.geom_size[-1, 1]
        length = sys.geom_size.at[-1, 1].set(cpole)
        return length, cpole

    length, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"geom_size": 0})
    sys = sys.tree_replace({"geom_size": length})
    return sys, in_axes, samples[:, None]


def uniform_state_sampler():
    pass
