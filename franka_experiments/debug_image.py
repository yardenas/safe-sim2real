import os

import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brax.training.agents.sac import checkpoint
from PIL import Image
from scipy.ndimage import binary_dilation

import ss2r.algorithms.sac.networks as sac_networks
import wandb
from ss2r.algorithms.sac.vision_networks import make_sac_vision_networks
from ss2r.common.wandb import get_wandb_checkpoint


def load_images(real_image, tiled_image):
    """Load and preprocess image from disk to match expected input."""
    img = Image.open(real_image).convert("RGB")
    img_array = np.asarray(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = Image.open(tiled_image)
    # Extract the upper left corner (0,0 to corner_size, corner_size)
    corner = img.crop((0, 0, 64, 64))
    return jnp.array(img_array), jnp.array(corner).astype(np.float32)[..., :3] / 255.0


def color_mask(image_np, lower, upper):
    return np.all((image_np >= lower) & (image_np <= upper), axis=-1)


def process_real_image(real_image):
    # Define color ranges
    red_lower = np.array([100, 0, 0]) / 255.0
    red_upper = np.array([255, 120, 120]) / 255.0
    white_lower = np.array([180, 180, 180]) / 255.0
    white_upper = np.array([255, 255, 255]) / 255.0
    black_lower = np.array([0, 0, 0]) / 255.0
    black_upper = np.array([20, 20, 20]) / 255.0
    turquoise_lower = np.array([0, 100, 100]) / 255  # dark-ish turquoise
    turquoise_upper = np.array([200, 255, 255]) / 255

    # Create masks
    mask_red = color_mask(real_image, red_lower, red_upper)
    mask_red = binary_dilation(mask_red, iterations=2)
    mask_white = color_mask(real_image, white_lower, white_upper)
    mask_white = binary_dilation(mask_white, iterations=2)
    # mask_white = binary_closing(mask_white, iterations=1)
    mask_black = color_mask(real_image, black_lower, black_upper)
    mask_black = binary_dilation(mask_black, iterations=2)
    mask_turquoise = color_mask(real_image, turquoise_lower, turquoise_upper)
    # Apply masks
    segmented = np.zeros_like(real_image)
    segmented[mask_red] = real_image[mask_red]
    segmented[mask_white] = real_image[mask_white]
    segmented[mask_black] = real_image[mask_black]
    segmented[mask_turquoise] = np.array([0, 0, 0])
    return segmented


def test_policy_outputs():
    api = wandb.Api()
    wandb_id = "eclz7est"
    run = api.run(f"ss2r/{wandb_id}")
    run_config = run.config
    restore_checkpoint_path = get_wandb_checkpoint(wandb_id, None)
    params = checkpoint.load(restore_checkpoint_path)
    act_size = 3
    obs_shape = (64, 64, 3)
    # Load actual image from disk
    current_dir = os.path.dirname(os.path.abspath(__file__))
    real_path = os.path.join(current_dir, "latest_image.png")
    sim_path = os.path.join(current_dir, "tiled_output.png")
    real_image, sim_image = load_images(real_path, sim_path)
    processed_image = process_real_image(real_image)
    activation = getattr(jnn, run_config["agent"]["activation"])
    sac_network = make_sac_vision_networks(
        observation_size={"pixels/view_0": obs_shape},
        action_size=act_size,
        policy_hidden_layer_sizes=run_config["agent"]["policy_hidden_layer_sizes"],
        value_hidden_layer_sizes=run_config["agent"]["value_hidden_layer_sizes"],
        encoder_hidden_dim=run_config["agent"]["encoder_hidden_dim"],
        activation=activation,
        tanh=run_config["agent"]["tanh"],
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    # Ensure inference function gets correct input format
    policy = make_policy((params[0], params[1]), True)
    obs = {"pixels/view_0": sim_image}
    jax_pred = policy(obs, jax.random.PRNGKey(0))[0]
    print("pred sim image", jax_pred)
    obs = {"pixels/view_0": real_image}
    jax_pred = policy(obs, jax.random.PRNGKey(0))[0]
    print("pred real image", jax_pred)
    obs = {"pixels/view_0": processed_image}
    jax_pred = policy(obs, jax.random.PRNGKey(0))[0]
    print("pred processed real image", jax_pred)
    _, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].imshow(sim_image)
    axes[0].axis("off")
    axes[0].set_title("Sim")
    axes[1].imshow(real_image)
    axes[1].axis("off")
    axes[1].set_title("Real")
    axes[2].imshow(processed_image)
    axes[2].axis("off")
    axes[2].set_title("Processed Real")
    plt.show()


if __name__ == "__main__":
    test_policy_outputs()
