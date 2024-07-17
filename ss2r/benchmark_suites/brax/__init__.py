import functools
from typing import Callable, Sequence
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from brax import envs
from brax.base import System
from brax.training.types import Policy, PRNGKey
from ss2r.benchmark_suites.brax.cartpole import cartpole
from ss2r.benchmark_suites.utils import get_task_config
from ss2r.rl.types import Simulator, SimulatorFactory, TrajectoryData, Transition


class BraxAdapter(Simulator):
    def __init__(
        self,
        environment: envs.PipelineEnv,
        seed: int,
        parallel_envs: int,
        randomization_fn: Callable[[System, PRNGKey], tuple[System, System, jax.Array]]
        | None,
        action_repeat: int,
    ) -> None:
        super().__init__()
        rng = jax.random.PRNGKey(seed)
        rng = jax.random.split(rng, parallel_envs)
        if randomization_fn is not None:
            new_sys, in_axes, samples = randomization_fn(environment.sys, rng)
            env = envs.training.wrap(
                environment,
                action_repeat=action_repeat,
                randomization_fn=lambda *_, **__: (new_sys, in_axes),
            )
            self.parameterizations = samples
        else:
            env = envs.training.wrap(environment, action_repeat=action_repeat)
        self.parallel_envs = parallel_envs
        self.environment = env

    @property
    def action_size(self) -> int:
        return self.environment.action_size

    @property
    def observation_size(self) -> int:
        return self.environment.observation_size

    def set_state(self, state: jax.Array) -> envs.State:
        q, qd = jnp.split(state, 2, axis=-1)
        assert q.shape[0] == qd.shape[0] == self.parallel_envs

        def set_env_state(sys, q, qd):
            env = self.environment._env_fn(sys=sys)
            state = env.pipeline_init(q, qd)
            obs = env._get_obs(state)
            reward, done, cost, steps, truncation = jnp.zeros(5)
            info = {
                "cost": cost,
                "steps": steps,
                "truncation": truncation,
                "first_pipeline_state": state,
                "first_obs": obs,
            }
            return envs.State(state, obs, reward, done, {}, info)

        res = jax.vmap(set_env_state, in_axes=(self.environment._in_axes, 0, 0))(
            self.environment._sys_v, q, qd
        )
        return res

    def step(
        self,
        env_state: envs.State,
        policy: Policy,
        key: PRNGKey,
        extra_fields: Sequence[str] = (),
    ) -> tuple[envs.State, Transition]:
        actions, policy_extras = policy(env_state.obs, key)
        nstate = self.environment.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        cost = state_extras.get("cost", jnp.zeros_like(nstate.reward))
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            cost=cost,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )

    def reset(self, seed: int | Sequence[int]) -> envs.State:
        if isinstance(seed, int):
            keys = jnp.asarray(
                jax.random.split(jax.random.PRNGKey(seed), self.parallel_envs)
            )
        else:
            assert len(seed) == self.parallel_envs
            keys = jnp.stack([jax.random.PRNGKey(s) for s in seed])
        return self.environment.reset(keys)

    def rollout(
        self,
        policy: Policy,
        steps: int,
        seed: int,
        state: envs.State | None = None,
    ) -> tuple[envs.State, TrajectoryData]:
        rng = jax.random.PRNGKey(seed)
        if state is None:
            rng, key = jax.random.split(rng)
            keys = jax.random.split(key, self.parallel_envs)
            state = self.environment.reset(keys)

        def f(carry, _):
            state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            nstate, transition = self.step(
                state,
                policy,
                current_key,
                extra_fields=("truncation",),
            )
            return (nstate, next_key), transition

        key = jax.random.PRNGKey(seed)
        (final_state, _), data = jax.lax.scan(f, (state, key), (), length=steps)
        return final_state, data


randomization_fns = {"inverted_pendulum": cartpole.domain_randomization}


def make(cfg: DictConfig) -> SimulatorFactory:
    def make_sim() -> BraxAdapter:
        task_cfg = get_task_config(cfg)
        env = envs.get_environment(task_cfg.task_name)
        if cfg.environment.brax.domain_randomization:
            randomize_fn = functools.partial(
                randomization_fns[task_cfg.task_name], cfg=task_cfg
            )
        else:
            randomize_fn = None
        sim = BraxAdapter(
            env,
            cfg.training.seed,
            cfg.training.parallel_envs,
            randomize_fn,
            cfg.training.action_repeat,
        )
        return sim

    return make_sim
