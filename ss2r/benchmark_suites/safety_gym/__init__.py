from io import BytesIO

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def add_text_to_frame(frame, text):
    fig, ax = plt.subplots(
        figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100
    )
    ax.imshow(frame)
    ax.axis("off")
    ax.text(
        10,
        25,
        text,
        color="white",
        fontsize=14,
        bbox=dict(facecolor="black", alpha=0.7, boxstyle="round,pad=0.3"),
    )
    # Render to memory
    canvas = FigureCanvas(fig)
    canvas.draw()
    # Save to memory buffer
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    # Convert to array (no disk involved)
    img = Image.open(buf).convert("RGB")
    img_array = np.array(img)
    plt.close(fig)
    return img_array


def render(env, policy, steps, rng, camera="fixedfar"):
    state = env.reset(rng)
    state = jax.tree_map(lambda x: x[:5], state)
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    videos = []
    orig_model = env._mjx_model
    for i in range(5):
        if hasattr(env, "_randomized_models"):
            model = jax.tree_map(
                lambda x, ax: jnp.take(x, i, axis=ax) if ax is not None else x,
                env._randomized_models,
                env._in_axes,
            )
        else:
            model = env._mjx_model
        ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        env._mjx_model = model
        video = env.render(ep_trajectory, camera=camera)
        rewards = np.asarray([step.reward for step in ep_trajectory])
        costs = np.asarray([step.info.get("cost", 0.0) for step in ep_trajectory])
        cum_rewards = np.cumsum(rewards)
        cum_costs = np.cumsum(costs)
        video_with_text = []
        for t, frame in enumerate(video):
            text = f"Reward: {cum_rewards[t]:.2f}  |  Cost: {cum_costs[t]:.2f}"
            frame_with_text = add_text_to_frame(frame, text)
            video_with_text.append(frame_with_text)
        videos.append(np.stack(video_with_text))
    env._mjx_model = orig_model
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)
