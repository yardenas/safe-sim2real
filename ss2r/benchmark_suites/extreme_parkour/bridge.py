import isaacgym
from brax.envs.base import Env, State, ObservationSize


import jax
import jax.dlpack
import jax.numpy as jnp

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    if isinstance(x, dict):
        return {k: torch2jax(v) for k, v in x.items()}
    elif x is None:
        return x
    elif x.dtype == torch.bool: # torch.bool is not supported by JAX's DLPack integration
        x = x.to(torch.int32)
    elif x.dtype in [torch.float16, torch.bfloat16, torch.float64]: # not necessarily needed but same problem for these types
        x = x.to(torch.float32)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))



class ExtremeParkourBridge(Env):

    def __init__(self, env):
        self._env = env
        
    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        action = jax2torch(action)

        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = self._env.step(action)

        obs_buf = torch2jax(obs_buf)
        rew_buf = torch2jax(rew_buf)
        reset_buf = torch2jax(reset_buf)
        print(extras)
        extras = {k: torch2jax(v) for k, v in extras.items()}

        return State(pipeline_state=None, obs=obs_buf, reward=rew_buf, done=reset_buf, info=extras)

    #The simulation is running 6144 agents in parallel
    def observation_size(self) -> ObservationSize:
        return 753 

    def action_size(self) -> int:
        return 12

    def backend(self) -> str:
        return "IsaacGym"


if __name__ == "__main__":
    args = get_args()
    args.headless = True
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env = ExtremeParkourBridge(env)
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    action = jax.random.uniform(rng, (6144, env.action_size()))

    for _ in range(10):
        state = env.step(state, action)
        print(f"Observation: {state.obs}, Reward: {state.reward}, Done: {state.done}")

print("Everything is working fine")
