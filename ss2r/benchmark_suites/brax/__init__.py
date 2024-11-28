import jax
import numpy as np
from brax.io import image

from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def render(env, policy, steps, rng):
    state = env.reset(rng)
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    videos = []
    for i in range(1):
        ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory.pipeline_state)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        video = image.render_array(env.sys, ep_trajectory)
        videos.append(video)
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)
