import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from ss2r.common.pytree import pytrees_unstack
from ss2r.rl.utils import rollout


def add_text_to_frame(frame, text, position=(10, 10)):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox(position, text)
    box_coords = [
        bbox[0] - 5,
        bbox[1] - 5,  # top-left
        bbox[2] + 5,
        bbox[3] + 5,  # bottom-right
    ]
    draw.rectangle(box_coords, fill=(0, 0, 0, 180))
    draw.text(position, text, fill=(255, 255, 255))
    return np.array(img)


def render(env, policy, steps, rng, camera="fixedfar"):
    state = env.reset(rng)
    state = jax.tree_map(lambda x: x[:5], state)
    orig_model = env._mjx_model
    model = jax.tree_map(
        lambda x, ax: jnp.take(x, jnp.arange(5), axis=ax) if ax is not None else x,
        env._randomized_models,
        env._in_axes,
    )
    env._mjx_model = model
    _, trajectory = rollout(env, policy, steps, rng[0], state)
    env._mjx_model = orig_model
    videos = []
    for i in range(5):
        ep_trajectory = jax.tree_map(lambda x: x[:, i], trajectory)
        ep_trajectory = pytrees_unstack(ep_trajectory)
        video = env.render(ep_trajectory, camera=camera)
        cum_rewards = cum_costs = 0
        video_with_text = []
        for t, frame in enumerate(video):
            cum_rewards += ep_trajectory[t].reward
            cum_costs += ep_trajectory[t].info.get("cost", 0)
            text = f"Reward: {cum_rewards:.2f}  |  Cost: {cum_costs:.2f}"
            frame_with_text = add_text_to_frame(frame, text)
            video_with_text.append(frame_with_text)
            if ep_trajectory[t].done:
                cum_rewards = cum_costs = 0
        videos.append(np.stack(video_with_text))
    return np.asarray(videos).transpose(0, 1, 4, 2, 3)
