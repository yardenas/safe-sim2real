import atexit
import functools
import logging
import multiprocessing as mp
import sys
import traceback
from collections.abc import Iterable
from enum import Enum

import cloudpickle
import numpy as np
from gymnasium.spaces import Box
from gymnasium.wrappers.clip_action import ClipAction
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.time_limit import TimeLimit

from ss2r.rl import wrappers

log = logging.getLogger("async_env")


class Protocol(Enum):
    ACCESS = 0
    CALL = 1
    RESULT = 2
    EXCEPTION = 3
    CLOSE = 4


# Based on
# https://github.com/danijar/dreamerv2/blob/07d906e9c4322c6fc2cd6ed23e247ccd6b7c8c41/dreamerv2/common/envs.py#L522
# as OpenAI gym's AsynVectorEnv fails to render nicely together with dm-control.
# (The main issue is with creating a dm-control
# https://github.com/openai/gym/blob/9a5db3b77a0c880ffed96ece1ab76eeff92c85e1/gym/vector/async_vector_env.py#L127
# which loads all the rendering handlers
# in the main process.)
class EpisodicAsync:
    def __init__(
        self,
        ctor,
        vector_size=1,
        time_limit=1000,
        action_repeat=1,
    ):
        self.env_fn = cloudpickle.dumps(ctor)
        self.time_limit = time_limit
        self.action_repeat = action_repeat
        self.parents = []
        self.processes = []
        for _ in range(vector_size):
            parent, process = self._make_worker()
            self.parents.append(parent)
            self.processes.append(process)
        atexit.register(self.close)
        for process in self.processes:
            process.start()
        self.observation_space = self.get_attr("observation_space")[0]
        self.action_space = self.get_attr("action_space")[0]
        self.num_envs = len(self.parents)

    def _make_worker(self):
        parent, child = mp.Pipe()
        process = mp.Process(
            target=_worker,
            args=(self.env_fn, child, self.time_limit, self.action_repeat),
        )
        return parent, process

    @functools.lru_cache
    def get_attr(self, name):
        for parent in self.parents:
            parent.send((Protocol.ACCESS, name))
        return self._receive()

    def close(self):
        try:
            for parent in self.parents:
                parent.send((Protocol.CLOSE, None))
                parent.close()
        except IOError:
            # The connection was already closed.
            pass
        for process in self.processes:
            process.join()

    def _receive(self, parents=None):
        payloads = []
        parents = parents or self.parents
        for parent in parents:
            try:
                message, payload = parent.recv()
            except ConnectionResetError:
                raise RuntimeError("Environment worker crashed.")
            # Re-raise exceptions in the main process.
            if message == Protocol.EXCEPTION:
                stacktrace = payload
                raise Exception(stacktrace)
            if message == Protocol.RESULT:
                payloads.append(payload)
            else:
                raise KeyError(f"Received message of unexpected type {message}")
        assert len(payloads) == len(parents)
        return payloads

    def step_async(self, actions):
        for parent, action in zip(self.parents, actions):
            payload = "step", (action,), {}  # type: ignore
            parent.send((Protocol.CALL, payload))

    def step_wait(self, **kwargs):
        observations, rewards, terminals, truncated, infos = zip(*self._receive())
        return (
            np.asarray(observations),
            np.asarray(rewards),
            np.asarray(truncated, dtype=bool) | np.asarray(terminals, dtype=bool),
            infos,
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        payload = name, args, kwargs
        for parent in self.parents:
            parent.send((Protocol.CALL, payload))

    def call_wait(self, **kwargs):
        return self._receive()

    def render(self):
        name = "render"
        args = ()
        kwargs = dict()
        payload = name, args, kwargs
        max_render = min(5, len(self.parents))
        for i in range(max_render):
            self.parents[i].send((Protocol.CALL, payload))
        return np.asarray(self._receive(self.parents[:max_render]))

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return self.reset_wait(seed, options)

    def reset_wait(
        self,
        seed=None,
        options=None,
    ):
        if seed is None:
            per_task_seed = [None] * self.num_envs
        elif isinstance(seed, int):
            per_task_seed = [seed + i for i in range(self.num_envs)]  # type: ignore
        else:
            per_task_seed = seed
        assert isinstance(per_task_seed, list) and len(per_task_seed) == self.num_envs
        if options is not None and "task" in options:
            assert isinstance(options["task"], Iterable)
            tasks = options.pop("task")
            assert len(tasks) == self.num_envs
        else:
            tasks = [None] * self.num_envs
        for parent, s, task in zip(self.parents, per_task_seed, tasks):
            if task is not None:
                assert options is not None
                task_options = options.copy()
                task_options["task"] = task
            else:
                task_options = options
            payload = (
                "reset",
                (),
                {"seed": s, "options": task_options},
            )
            parent.send((Protocol.CALL, payload))
        outs = np.asarray([x[0] for x in self.call_wait()])
        return outs


def _worker(ctor, conn, time_limit, action_repeat):
    try:
        env = TimeLimit(cloudpickle.loads(ctor)(), time_limit)
        if isinstance(env.action_space, Box):
            env = RescaleAction(env, -1.0, 1.0)  # type: ignore
            env.action_space = Box(-1.0, 1.0, env.action_space.shape, np.float32)
            env = ClipAction(env)  # type: ignore
        env = wrappers.ActionRepeat(env, action_repeat)  # type: ignore
        while True:
            try:
                # Only block for short times to have keyboard exceptions be raised.
                if not conn.poll(0.1):
                    continue
                message, payload = conn.recv()
            except (EOFError, KeyboardInterrupt):
                break
            if message == Protocol.ACCESS:
                name = payload
                try:
                    result = getattr(env, name)
                except AttributeError:
                    result = None
                conn.send((Protocol.RESULT, result))
                continue
            if message == Protocol.CALL:
                name, args, kwargs = payload
                result = getattr(env, name)(*args, **kwargs)
                conn.send((Protocol.RESULT, result))
                continue
            if message == Protocol.CLOSE:
                assert payload is None
                break
            raise KeyError(f"Received message of unknown type {message}")
    except Exception:
        stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
        log.error(f"Error in environment process: {stacktrace}")
        conn.send((Protocol.EXCEPTION, stacktrace))
    finally:
        conn.close()
