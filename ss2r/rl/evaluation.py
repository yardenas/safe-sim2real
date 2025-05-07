import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from brax.envs.wrappers.training import EvalMetrics, EvalWrapper
from brax.training.acting import Evaluator, generate_unroll
from brax.training.types import Metrics, Policy, PolicyParams, PRNGKey


class ConstraintEvalWrapper(EvalWrapper):
    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        reset_state.metrics["cost"] = reset_state.info.get(
            "cost", jnp.zeros_like(reset_state.reward)
        )
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jnp.zeros_like, reset_state.metrics),
            active_episodes=jnp.ones_like(reset_state.reward),
            episode_steps=jnp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        nstate.metrics["cost"] = nstate.info.get("cost", jnp.zeros_like(nstate.reward))
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info.get("steps", jnp.zeros_like(state_metrics.episode_steps)),
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)
        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info["eval_metrics"] = eval_metrics
        return nstate


class ConstraintsEvaluator(Evaluator):
    def __init__(
        self,
        eval_env: Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],  # type: ignore
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: jax.Array,
        budget: float,
        num_episodes: int = 10,
    ):
        self._key = key
        self._eval_walltime = 0.0
        eval_env = ConstraintEvalWrapper(eval_env)
        self.budget = budget
        self.num_episodes = num_episodes

        def generate_eval_unroll(policy_params: PolicyParams, key: PRNGKey) -> State:  # type: ignore
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
            )[0]

        self._generate_eval_unroll = jax.jit(
            jax.vmap(generate_eval_unroll, in_axes=(None, 0))
        )
        self._steps_per_unroll = episode_length * num_eval_envs * num_episodes

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)
        unroll_key = jax.random.split(unroll_key, self.num_episodes)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        constraint = eval_state.info["eval_metrics"].episode_metrics["cost"].mean(0)
        eval_state.info["eval_metrics"].episode_metrics["cost"] = constraint
        safe = np.where(constraint < self.budget, 1.0, 0.0)
        eval_state.info["eval_metrics"].episode_metrics["safe"] = safe
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = "_std" if fn == np.std else ""
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(value) if aggregate_episodes else value  # type: ignore
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            "eval/walltime": self._eval_walltime,
            **training_metrics,
            **metrics,
        }
        return metrics
