import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.go1_sac_to_onnx import convert_policy_to_onnx
from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.online import OnlineEpisodeOrchestrator
from ss2r.rl.types import MakePolicyFn, UnrollFn


def get_collection_fn(cfg):
    if cfg.agent.data_collection.name == "step":
        return collect_single_step
    elif cfg.agent.data_collection.name == "episodic":
        fn = (
            lambda env,
            env_state,
            make_policy_fn,
            policy_params,
            key,
            extra_fields: generate_unroll(
                env,
                env_state,
                make_policy_fn,
                policy_params,
                key,
                cfg.training.episode_length,
                extra_fields,
            )
        )
        return make_collection_fn(fn)
    elif cfg.agent.data_collection.name == "hardware":
        data_collection_cfg = cfg.agent.data_collection
        if "Go1" in cfg.environment.task_name:
            policy_translate_fn = functools.partial(make_go1_policy, cfg=cfg)
        else:
            raise ValueError(
                f"Environment {cfg.environment.task_name} not supported for hardware data collection."
            )
        orchestrator = OnlineEpisodeOrchestrator(
            policy_translate_fn,
            cfg.training.episode_length,
            data_collection_cfg.address,
        )
        return make_collection_fn(orchestrator.request_data)
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
        if transitions.reward.ndim == 2:
            transitions = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
        normalizer_params = running_statistics.update(
            normalizer_params, transitions.observation
        )
        buffer_state = replay_buffer.insert(buffer_state, float16(transitions))
        return normalizer_params, env_state, buffer_state

    return collect_data


collect_single_step = make_collection_fn(actor_step)


def make_go1_policy(make_policy_fn, params, cfg):
    del make_policy_fn
    proto_model = convert_policy_to_onnx(params, cfg, 12, 48)
    return proto_model.SerializeToString()
