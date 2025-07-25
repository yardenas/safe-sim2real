import os

import jax
import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from brax.training.agents.sac import checkpoint
from PIL import Image

import ss2r.algorithms.sac.networks as sac_networks
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
    obs = {"pixels/view_0": real_image}
    jax_pred = policy(obs, jax.random.PRNGKey(0))[0]
    print("pred real image", jax_pred)
    obs = {"pixels/view_0": sim_image}
    jax_pred = policy(obs, jax.random.PRNGKey(0))[0]
    print("pred sim image", jax_pred)
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(real_image)
    axes[0].axis("off")
    axes[0].set_title("Real")
    axes[1].imshow(sim_image)
    axes[1].axis("off")
    axes[1].set_title("Sim")
    plt.show()


if __name__ == "__main__":
    test_policy_outputs()
