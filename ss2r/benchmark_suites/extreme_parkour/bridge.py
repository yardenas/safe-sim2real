import isaacgym  # noqa
import jax
import jax.dlpack
import torch
from brax.envs.base import Env, State
import legged_gym.envs  # noqa


def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


def torch2jax(x):
    if isinstance(x, dict):
        return {k: torch2jax(v) for k, v in x.items()}
    elif x is None:
        return x
    elif (
        x.dtype == torch.bool
    ):  # torch.bool is not supported by JAX's DLPack integration
        x = x.to(torch.int32)
    elif x.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float64,
    ]:  # not necessarily needed but same problem for these types
        x = x.to(torch.float32)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))


class ExtremeParkourBridge(Env):
    def __init__(self, env):
        self.env = env

    def reset(self, rng: jax.Array) -> State:
        pass

    def step(self, state: State, action: jax.Array) -> State:
        action = jax2torch(action)

        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = self.env.step(action)

        obs_buf = torch2jax(obs_buf)
        rew_buf = torch2jax(rew_buf)
        reset_buf = torch2jax(reset_buf)
        extras = torch2jax(extras)

        # TODO check with Chenhao if this is correct (do we need the forces and what is rigid body state?)
        pipeline_state = {
            "actor_root": self.env.gym.aquire_actor_root_state_tensor(self.env.sim),
            "dof_state": self.env.gym.acquire_dof_state_tensor(self.env.sim),
            "dof_force": self.env.gym.acquire_dof_force_tensor(self.env.sim),
            "rigid_body_state": self.env.gym.acquire_rigid_body_state_tensor(
                self.env.sim
            ),
        }

        return State(
            pipeline_state=pipeline_state,
            obs=obs_buf,
            reward=rew_buf,
            done=reset_buf,
            info=extras,
        )

    # The simulation is running 6144 agents in parallel
    def observation_size(self) -> int:
        return 753

    def action_size(self) -> int:
        return 12

    def backend(self) -> str:
        return "IsaacGym"
