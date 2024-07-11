from typing import Callable, Sequence
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
from brax import envs
from brax.base import System
from brax.training.types import Policy, PRNGKey
from ss2r.benchmark_suites.brax.cartpole import domain_randomization
from ss2r.benchmark_suites.utils import get_domain_and_task
from ss2r.rl.trajectory import TrajectoryData, Transition
from ss2r.rl.types import Simulator, SimulatorFactory


class BraxAdapter(Simulator):
    def __init__(
        self,
        environment: envs.PipelineEnv,
        seed: int,
        parallel_envs: int,
        randomization_fn: Callable[[System, PRNGKey], tuple[System, System, jax.Array]],
        action_repeat: int,
    ) -> None:
        super().__init__()
        rng = jax.random.PRNGKey(seed)
        rng = jax.random.split(rng, parallel_envs)
        new_sys, in_axes, samples = randomization_fn(environment.sys, rng)
        env = envs.training.wrap(
            environment,
            action_repeat=action_repeat,
            randomization_fn=lambda *_: (new_sys, in_axes),
        )
        self.parallel_envs = parallel_envs
        self.environment = env
        self.parameterizations = samples

    @property
    def action_size(self) -> int:
        return self.environment.action_size

    @property
    def observation_size(self) -> int:
        return self.environment.observation_size

    def set_state(self, state: jax.Array) -> envs.State | None:
        q, qd = jnp.split(state, 2, axis=1)
        state = self.environment.pipeline_init(q, qd)
        return state

    def step(
        self,
        env_state: envs.State,
        policy: Policy,
        key: PRNGKey,
        extra_fields: Sequence[str] = (),
    ) -> Transition:
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

    def reset(self, seed: int) -> envs.State:
        key = jnp.asarray(
            jax.random.split(jax.random.PRNGKey(seed), self.parallel_envs)
        )
        return self.environment.reset(key)

    def rollout(
        self,
        policy: Policy,
        steps: int,
        state: envs.State | None = None,
        key: PRNGKey | None = None,
    ) -> TrajectoryData:
        if key is None:
            # TODO (yarden): fix this
            key = jax.random.PRNGKey(0)
        if state is None:
            key, subkey = jax.random.split(key)
            subkey = jax.random.split(subkey, self.parallel_envs)
            state = self.reset(666)

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

        (final_state, _), data = jax.lax.scan(f, (state, key), (), length=steps)
        return final_state, data


randomization_fns = {"inverted_pendulum": domain_randomization}


def make(cfg: DictConfig) -> SimulatorFactory:
    def make_sim():
        _, task_cfg = get_domain_and_task(cfg)
        env = envs.get_environment(task_cfg.task)
        sim = BraxAdapter(
            env,
            cfg.training.seed,
            cfg.training.parallel_envs,
            randomization_fns[task_cfg.task],
            cfg.training.action_repeat,
        )
        return sim

    return make_sim  # type: ignore
