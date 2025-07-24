import os

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import wandb
from brax.training.agents.sac import checkpoint
from PIL import Image

import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac.vision_networks import make_sac_vision_networks
from ss2r.common.wandb import get_wandb_checkpoint


def load_image(image_path):
    """Load and preprocess image from disk to match expected input."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.asarray(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return jnp.array(img_array)


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
    image_path = os.path.join(current_dir, "latest_image.png")
    obs_img = load_image(image_path)
    obs_img = load_image(image_path)
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
    batched_obs = {
        "pixels/view_0": jnp.expand_dims(obs_img, axis=0)
    }  # Add batch dimension
    policy = make_policy((params[0], params[1]), True)
    action = policy(batched_obs, jax.random.PRNGKey(0))[0]
    print("Action:", action)


if __name__ == "__main__":
    test_policy_outputs()
