import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey, Transition

from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.types import MakePolicyFn, UnrollFn


def get_collection_fn(cfg):
    if cfg.agent.data_collection.name == "step":
        return collect_single_step
    elif cfg.agent.data_collection.name == "episodic":

        def generate_episodic_unroll(
            env,
            env_state,
            make_policy_fn,
            policy_params,
            key,
            extra_fields,
        ):
            env_state, transitions = acting.generate_unroll(
                env,
                env_state,
                make_policy_fn(policy_params),
                key,
                cfg.training.episode_length,
                extra_fields,
            )
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
            return env_state, transitions

        return make_collection_fn(generate_episodic_unroll)
    elif cfg.agent.data_collection.name == "hardware":
        data_collection_cfg = cfg.agent.data_collection
        if "Go1" in cfg.environment.task_name or "Go2" in cfg.environment.task_name:
            from ss2r.algorithms.sac.go1_sac_to_onnx import (
                go1_postprocess_data,
                make_go1_policy,
            )
            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = functools.partial(make_go1_policy, cfg=cfg)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                go1_postprocess_data,
                data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        elif "rccar" in cfg.environment.task_name:
            import cloudpickle

            from ss2r.rl.online import OnlineEpisodeOrchestrator

            policy_translate_fn = lambda _, params: cloudpickle.dumps(params)
            orchestrator = OnlineEpisodeOrchestrator(
                policy_translate_fn,
                cfg.training.episode_length,
                data_collection_cfg.wait_time_sec,
                lambda data: Transition(*data),
                address=data_collection_cfg.address,
            )
            return make_collection_fn(orchestrator.request_data)
        else:
            raise ValueError(
                f"Environment {cfg.environment.task_name} not supported for hardware data collection."
            )
    else:
        raise ValueError(f"Unknown data collection {cfg.agent.data_collection.name}")


def actor_step(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.actor_step(env, env_state, policy, key, extra_fields)


def generate_unroll(
    env,
    env_state,
    make_policy_fn,
    policy_params,
    key,
    unroll_length,
    extra_fields,
):
    policy = make_policy_fn(policy_params)
    return acting.generate_unroll(
        env, env_state, policy, key, unroll_length, extra_fields
    )


def make_collection_fn(unroll_fn: UnrollFn) -> CollectDataFn:
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn,
        params: Params,
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        env_state, transitions = unroll_fn(
            env,
            env_state,
            make_policy_fn,
            (normalizer_params, params),
            key,
            extra_fields=extra_fields,
        )
        normalizer_params = running_statistics.update(
            normalizer_params, transitions.observation
        )
        buffer_state = replay_buffer.insert(buffer_state, float16(transitions))
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(actor_step)
