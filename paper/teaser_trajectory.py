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
disagreement_values = []
for i in range(1):
    ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
    ep_trajectory = pytrees_unstack(ep_trajectory)
    video = env.render(ep_trajectory)
    videos.append(video)
    ep_disagreement = np.array(trajectory.info["disagreement"])[:, i]
    disagreement_values.append(ep_disagreement)
frames = np.asarray(videos)
disagreement_values = np.asarray(disagreement_values)  # type: ignore

# %%
fig, ax = plt.subplots()
im = ax.imshow(frames[0, 0])
text = ax.text(10, 10, "", fontsize=12, color="red", backgroundcolor="white")


def update(frame_idx):
    frame = frames[0, frame_idx]
    im.set_array(frame)
    # Scale disagreement for visualization (optional)
    disagreement = disagreement_values[0, frame_idx]
    text.set_text(f"Disagreement: {disagreement:.2f}")
    return [im, text]


ani = animation.FuncAnimation(
    fig, update, frames=frames.shape[1], interval=50, blit=True
)
plt.show()
