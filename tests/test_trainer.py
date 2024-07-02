import pathlib
import time
import pytest
from tests import DummyAgent, make_test_config
from ss2r.rl.trainer import Trainer
from ss2r import benchmark_suites


@pytest.fixture
def config():
    cfg = make_test_config(["training.action_repeat=4"])
    return cfg


@pytest.fixture
def trainer(config):
    make_env = benchmark_suites.make(config)
    dummy_env = make_env()
    with Trainer(
        config,
        make_env,
        DummyAgent(dummy_env.action_size, config),
    ) as trainer:
        yield trainer
    assert trainer.state_writer is not None
    pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").unlink()


def test_epoch(trainer):
    trainer.train(1)
    wait_count = 10
    while wait_count > 0:
        time.sleep(0.5)
        if not pathlib.Path(f"{trainer.state_writer.log_dir}/state.pkl").exists():
            wait_count -= 1
            if wait_count == 0:
                pytest.fail("state file was not written")
        else:
            break
    new_trainer = Trainer.from_pickle(
        trainer.config, f"{trainer.state_writer.log_dir}/state.pkl"
    )
    assert new_trainer.step == trainer.step
    assert new_trainer.epoch == trainer.epoch
    assert new_trainer.seeds is not None
    assert (new_trainer.seeds.key == trainer.seeds.key).all()
    with new_trainer as new_trainer:
        new_trainer_summary, *_ = new_trainer._run_training_epoch()
    old_trainer_summary, *_ = trainer._run_training_epoch()
    assert old_trainer_summary.metrics == new_trainer_summary.metrics
