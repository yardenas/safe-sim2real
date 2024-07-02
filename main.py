import logging

import hydra
import jax
from omegaconf import OmegaConf

from ss2r.rl.trainer import get_state_path, load_state, should_resume, start_fresh

_LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="config")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    state_path = get_state_path()
    if should_resume(state_path):
        _LOG.info(f"Resuming experiment from: {state_path}")
        trainer = load_state(cfg, state_path)
    else:
        _LOG.info("Starting a new experiment.")
        trainer = start_fresh(cfg)
    with trainer, jax.disable_jit(not cfg.jit):
        trainer.train()
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
