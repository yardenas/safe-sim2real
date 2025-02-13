# Installation Guide for Extreme Parkour

## Setup Environment

```sh
# Create conda environment with Python 3.8
conda create -n parkour python=3.8
conda activate parkour
```

## Install Dependencies

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121/torch_stable.html
```

## Install Isaac Gym and Modified Extreme Parkour Repository
Clone the repository
```sh
cd ~
git clone git@github.com:bungeru/safe-extreme-parkour-RL.git
cd safe-extreme-parkour-RL
```
Install packages
```sh
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym
# Originally trained with Preview3, but haven't seen bugs using Preview4.
cd isaacgym/python && pip install -e .
cd ~/safe-extreme-parkour-RL/rsl_rl && pip install -e .
cd ~/safe-extreme-parkour-RL/legged_gym && pip install -e .
```

## Additional Dependencies

```sh
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

## Possibly change Environment Variables

```sh
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:~/miniconda3/envs/parkour/lib
```

## Install Safe Sim2Real

```sh
cd ~
git clone https://github.com/yardenas/safe-sim2real.git
pip install --upgrade "jaxlib[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade jax
pip install brax chex jaxopt optax orbax-checkpoint mujoco-mjx hydra-core
```

## Usage

### Running Training

Run the training by calling `train_brax.py` with the correct config file, here `extreme_parkour`.

```sh
python train_brax.py environment.task_name=extreme_parkour
```

