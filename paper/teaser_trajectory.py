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


def split_trajectories(trajectory):
    done_array = np.array(trajectory.done)
    trajectories = []
    for i in range(1):
        indices = np.where(done_array[:, i] == 1)[0]
        # Add the first index (0) at the beginning to capture the initial segment
        split_indices = np.concatenate(([0], indices + 1))
        for j in range(len(split_indices) - 1):
            # Extract sub-trajectory
            start, end = split_indices[j], split_indices[j + 1]
            sub_trajectory = jax.tree_map(lambda x: x[start:end, i], trajectory)
            trajectories.append(sub_trajectory)
        # Add the remaining part of the trajectory after the last done=True, if any
        if split_indices[-1] < done_array.shape[0]:
            sub_trajectory = jax.tree_map(
                lambda x: x[split_indices[-1] :, i], trajectory
            )
            trajectories.append(sub_trajectory)
    return trajectories


flattened_trajectories = split_trajectories(trajectory)

for trajectory in flattened_trajectories:
    ep_trajectory = pytrees_unstack(trajectory)
    video = env.render(ep_trajectory)
    ep_disagreement = np.array(trajectory.info["disagreement"])
    frames = np.asarray(video)
    disagreement_values = np.asarray(ep_disagreement)
    fig, (ax_frame, ax_plot) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    im = ax_frame.imshow(frames[0])
    ax_frame.axis("off")  # Hide the axis for the frame display
    # Plot the initial disagreement line plot on the right
    ax_plot.set_xlim(0, frames.shape[0])
    ax_plot.set_ylim(np.min(disagreement_values), np.max(disagreement_values))
    (line,) = ax_plot.plot([], [], color="blue")
    ax_plot.set_title("Disagreement Over Time")
    ax_plot.set_xlabel("Frame ID")
    ax_plot.set_ylabel("Disagreement")
    x_data, y_data = [], []  # type: ignore

    def update(frame_idx):
        frame = frames[frame_idx]
        im.set_array(frame)
        # Update disagreement plot
        if trajectory.done[frame_idx]:
            x_data.clear()
            y_data.clear()
        else:
            x_data.append(frame_idx)
            y_data.append(disagreement_values[frame_idx])
        line.set_data(x_data, y_data)
        ax_plot.set_xlim(0, frames.shape[0])  # Adjust x-axis limit dynamically
        return [im, line]

    ani = animation.FuncAnimation(
        fig, update, frames=frames.shape[0], interval=50, blit=True
    )
    plt.tight_layout()
    plt.show()
