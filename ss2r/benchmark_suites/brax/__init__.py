import jax
from brax.io import image

from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def render(env, policy, steps, rng):
    _, trajectory = rollout(policy, steps, rng)
    trajectory = jax.tree_map(lambda x: x[:, 0], trajectory.extras["pipeline_state"])
    trajectory = pytrees_unstack(trajectory)
    video = image.render_array(env.sys, trajectory)
    return video
