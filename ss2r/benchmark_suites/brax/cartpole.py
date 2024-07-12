import jax


def domain_randomization(sys, rng):
    @jax.vmap
    def randomize(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-5.1, maxval=5.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

    samples = randomize(rng)
    sys_v = sys.tree_replace({'link.inertia.transform.pos': samples})
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({'link.inertia.transform.pos': 0})
    return sys_v, in_axes, samples


def uniform_state_sampler():
    pass
