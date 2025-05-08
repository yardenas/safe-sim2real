# safe-sim2real

# PTSD: Provably Safe Transfer via Domain Randomization 🛡️🤖

**Bridging the Sim-to-Real Gap in Safety-Critical Reinforcement Learning**

Deploying reinforcement learning (RL) in the real world—particularly in safety-critical domains—remains a daunting challenge due to the inherent risks of unsafe exploration. While **domain randomization** has shown promise in transferring policies from simulation to reality, it lacks formal safety guarantees.

**PTSD** (Provably Safe Transfer via Domain Randomization) introduces a **theoretically grounded** approach that ensures **constraint satisfaction** even in the presence of **model mismatch** and **domain shift**, relying *only* on domain randomization. Our method maintains the **simplicity and scalability** of existing sim-to-real paradigms while providing provable safety during real-world deployment.
---
## Requirements 🛠

- **Python** ≥ 3.10
- Recommended environment managers: `venv` or `Poetry`

## Installation 🧩

### Using pip

```bash
git clone https://github.com/yardenas/safe-sim2real
cd safe-sim2real
python3 -m venv venv
source venv/bin/activate
pip install -e .
````

### Using Poetry

```bash
git clone https://github.com/yardenas/safe-sim2real
cd safe-sim2real
poetry install
poetry shell
```

## Usage 🧪

Our extensively uses [Hydra](https://hydra.cc/) to configure experiments. Each experiment is defined as a `yaml` file in `ss2r/configs/experiments`. For example, to train a Unitree Go1 policy with a constraint on joint limit:

```bash
python train_brax.py +experiment=go1_sim_to_real
```


<!-- ## Citation 🔗

If you find our repository useful in your work, please consider citing:

```bibtex
@inproceedings{your2025ptsd,
  title={PTSD: Provably Safe Transfer via Domain Randomization},
  author={Your Name and Collaborators},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2025},
  url={https://openreview.net/forum?id=XXXX}
}
``` -->

<!-- ## Learn More 🔍

* **Project Webpage**: [https://yourpage.github.io/ptsd](https://yourpage.github.io/ptsd)
* **Paper**: \[arXiv/CoRL link here]
* **Contact**: For support or feedback, please open an issue or reach out via the webpage. -->

