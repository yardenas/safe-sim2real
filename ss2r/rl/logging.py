import json
import logging
import os
from queue import Queue
from threading import Thread
from typing import Any, Protocol

import cloudpickle
import numpy as np
import omegaconf
from numpy import typing as npt
from omegaconf import DictConfig
from omegaconf.errors import InterpolationKeyError
from tabulate import tabulate

_SUMMARY_DEFAULT = "summary"


class Writer(Protocol):
    def log(self, summary: dict[str, float], step: int):
        ...

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        ...


class TrainingLogger:
    def __init__(self, config: DictConfig) -> None:
        self._writers: list[Writer] = []
        log_path = os.getcwd()
        for writer in config.writers:
            if writer == "wandb":
                self._writers.append(WeightAndBiasesWriter(config))
            elif writer == "jsonl":
                self._writers.append(JsonlWriter(log_path))
            elif writer == "tensorboard":
                self._writers.append(TensorboardXWriter(log_path))
            elif writer == "stderr":
                self._writers.append(StdErrWriter())
            else:
                raise ValueError(f"Unknown writer: {writer}")

    def log(self, summary: dict[str, float], step: int):
        for writer in self._writers:
            writer.log(summary, step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        for writer in self._writers:
            writer.log_video(images, step, name, fps)


class StdErrWriter:
    def __init__(self, logger_name: str = _SUMMARY_DEFAULT):
        self._logger = logging.getLogger(logger_name)

    def log(self, summary: dict[str, float], step: int):
        to_log = [[k, v] for k, v in summary.items()]
        self._logger.info(
            f"Step {step} summary:\n"
            + tabulate(to_log, headers=["Metric", "Value"], tablefmt="orgtbl")
        )

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        pass


class JsonlWriter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

    def log(self, summary: dict[str, float], step: int):
        with open(os.path.join(self.log_dir, f"{_SUMMARY_DEFAULT}.jsonl"), "a") as file:
            file.write(json.dumps({"step": step, **summary}) + "\n")

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        pass


class TensorboardXWriter:
    def __init__(self, log_dir) -> None:
        import tensorboardX

        self._writer = tensorboardX.SummaryWriter(log_dir)

    def log(self, summary: dict[str, float], step: int):
        for k, v in summary.items():
            self._writer.add_scalar(k, float(v), step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
        flush: bool = False,
    ):
        self._writer.add_video(name, np.array(images, copy=False), step, fps=fps)
        if flush:
            self._writer.flush()


class WeightAndBiasesWriter:
    def __init__(self, config: DictConfig):
        import wandb

        try:
            name = config.wandb.name
        except InterpolationKeyError:
            name = None
        config.wandb.name = name
        config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
        assert isinstance(config_dict, dict)
        wandb.init(project="safe-opax", resume=True, config=config_dict, **config.wandb)
        self._handle = wandb

    def log(self, summary: dict[str, float], step: int):
        self._handle.log(summary, step=step)

    def log_video(
        self,
        images: npt.ArrayLike,
        step: int,
        name: str = "policy",
        fps: int | float = 30,
    ):
        self._handle.log(
            {
                name: self._handle.Video(
                    np.array(images, copy=False),
                    fps=int(fps),
                    caption=name,
                )
            },
            step=step,
        )


class StateWriter:
    def __init__(self, log_dir: str, state_filename: str):
        self.log_dir = log_dir
        self.state_filename = state_filename
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.queue: Queue[bytes] = Queue(maxsize=5)
        self._thread = Thread(name="state_writer", target=self._worker)
        self._thread.start()

    def write(self, data: dict[str, Any]):
        state_bytes = cloudpickle.dumps(data)
        self.queue.put(state_bytes)
        # Lazily open up a thread and let it drain the work queue. Thread exits
        # when there's no more work to do.
        if not self._thread.is_alive():
            self._thread = Thread(name="state_writer", target=self._worker)
            self._thread.start()

    def _worker(self):
        while not self.queue.empty():
            state_bytes = self.queue.get(timeout=1)
            with open(os.path.join(self.log_dir, self.state_filename), "wb") as f:
                f.write(state_bytes)
                self.queue.task_done()

    def close(self):
        self.queue.join()
        if self._thread.is_alive():
            self._thread.join()
