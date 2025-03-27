# %%
import pickle

import jax
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mujoco_playground import registry

from ss2r.common.pytree import pytrees_unstack

# %%

with open("./data/trajectory.pkl", "rb") as f:
    trajectory = pickle.load(f)


# %%

env_name = "Go1JoystickFlatTerrain"
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)
videos = []
for i in range(5):
    ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
    ep_trajectory = pytrees_unstack(ep_trajectory)
    video = env.render(ep_trajectory)
    videos.append(video)
frames = np.asarray(videos).transpose(0, 1, 4, 2, 3)

# %%
fig, ax = plt.subplots()
im = ax.imshow(frames[0])


def update(frame):
    im.set_array(frame)
    return [im]


ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
plt.show()
