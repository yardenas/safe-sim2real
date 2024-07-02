import logging
import os
import time
from typing import Optional

import cloudpickle
from omegaconf import DictConfig

from ss2r import benchmark_suites
from ss2r.rl import acting
from ss2r.rl.epoch_summary import EpochSummary
from ss2r.rl.logging import StateWriter, TrainingLogger
from ss2r.rl.types import Agent, SimulatorFactory, Simulator
from ss2r.rl.utils import PRNGSequence

_LOG = logging.getLogger(__name__)

_TRAINING_STATE = "state.pkl"


def get_state_path() -> str:
    log_path = os.getcwd()
    state_path = os.path.join(log_path, _TRAINING_STATE)
    return state_path


def should_resume(state_path: str) -> bool:
    return os.path.exists(state_path)


def start_fresh(
    cfg: DictConfig,
) -> "Trainer":
    make_env = benchmark_suites.make(cfg)
    return Trainer(cfg, make_env)


def load_state(cfg, state_path) -> "Trainer":
    return Trainer.from_pickle(cfg, state_path)


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        make_sim: SimulatorFactory,
        agent: Agent | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
    ):
        self.config = config
        self.make_sim = make_sim
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: TrainingLogger | None = None
        self.state_writer: StateWriter | None = None
        self.simulator: Simulator | None = None
        self.agent = agent

    def __enter__(self):
        log_path = os.getcwd()
        self.logger = TrainingLogger(self.config)
        self.state_writer = StateWriter(log_path, _TRAINING_STATE)
        self.simulator = self.make_sim()
        if self.seeds is None:
            self.seeds = PRNGSequence(self.config.training.seed)
        if self.agent is None:
            self.agent = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()

    def train(self, epochs: Optional[int] = None) -> None:
        epoch, logger, state_writer, agent = (
            self.epoch,
            self.logger,
            self.state_writer,
            self.agent,
        )
        assert logger is not None and state_writer is not None and agent is not None
        for epoch in range(epoch, epochs or self.config.training.epochs):
            _LOG.info(f"Training epoch #{epoch}")
            report, wall_time = self._run_training_epoch()
            summary = self.evaluate()
            objective, cost_rate, feasibilty = summary.metrics
            metrics = {
                "train/objective": objective,
                "train/cost_rate": cost_rate,
                "train/feasibility": feasibilty,
                "train/fps": self.config.training.steps_per_epoch / wall_time,
            }
            report.metrics.update(metrics)
            logger.log(report.metrics, self.step)
            for k, v in report.videos.items():
                logger.log_video(v, self.step, k)
            self.epoch = epoch + 1
            state_writer.write(self.state)

    def _run_training_epoch(self) -> tuple[EpochSummary, float, int]:
        agent, sim, logger, seeds = self.agent, self.simulator, self.logger, self.seeds
        assert (
            sim is not None
            and agent is not None
            and logger is not None
            and seeds is not None
        )
        start_time = time.time()
        sim.reset(seed=int(next(seeds)[0].item()))
        report = agent.train(self.config.training.steps_per_epoch, sim)
        self.step += self.config.training.steps_per_epoch
        next(seeds)
        end_time = time.time()
        wall_time = end_time - start_time
        return report, wall_time

    def evaluate(self) -> EpochSummary:
        agent, sim, logger, seeds = self.agent, self.simulator, self.logger, self.seeds
        assert (
            sim is not None
            and agent is not None
            and logger is not None
            and seeds is not None
        )
        sim.reset(seed=int(next(seeds)[0].item()))
        summary = acting.evaluate(
            agent,
            sim,
            self.config.training.num_eval_steps,
            self.config.training.render_episodes,
        )
        return summary

    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "Trainer":
        with open(state_path, "rb") as f:
            make_sim, seeds, agent, epoch, step = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        _LOG.info(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_sim=make_sim,
            start_epoch=epoch,
            seeds=seeds,
            agent=agent,
            step=step,
        )

    @property
    def state(self):
        return {
            "make_sim": self.make_sim,
            "seeds": self.seeds,
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
        }
