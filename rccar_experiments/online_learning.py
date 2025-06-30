import hydra
import jax

from rccar_experiments.experiment_driver import ExperimentDriver
from ss2r.benchmark_suites.rccar import hardware


@hydra.main(
    version_base=None,
    config_path="../ss2r/configs",
    config_name="rccar_online_learning",
)
def main(cfg):
    with (
        hardware.connect(
            car_id=cfg.car_id,
            port_number=cfg.port_number,
            control_frequency=cfg.control_frequency,
        ) as controller,
        jax.disable_jit(),
    ):
        driver = ExperimentDriver(cfg, controller)
        driver.run()


if __name__ == "__main__":
    main()
