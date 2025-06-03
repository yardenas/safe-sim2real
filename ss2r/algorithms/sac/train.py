# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Soft Actor-Critic training.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.training import replay_buffers
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint
from brax.training.types import Params, PRNGKey
from ml_collections import config_dict

import ss2r.algorithms.sac.losses as sac_losses
import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.penalizers import Penalizer
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.data import collect_single_step
from ss2r.algorithms.sac.q_transforms import QTransformation, SACBase, SACCost, UCBCost
from ss2r.algorithms.sac.rae import RAEReplayBuffer
from ss2r.algorithms.sac.training_step import make_training_step
from ss2r.algorithms.sac.types import (
    CollectDataFn,
    Metrics,
    ReplayBufferState,
    TrainingState,
    Transition,
    float16,
)
from ss2r.rl.evaluation import ConstraintsEvaluator


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    sac_network: sac_networks.SafeSACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    qr_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation | None,
    penalizer_params: Params | None,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_qr, key_qc = jax.random.split(key, 3)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    qr_params = sac_network.qr_network.init(key_qr)
    if sac_network.qc_network is not None:
        qc_params = sac_network.qc_network.init(key_qr)
        assert qc_optimizer is not None
        qc_optimizer_state = qc_optimizer.init(qc_params)
    else:
        qc_params = None
        qc_optimizer_state = None
    qr_optimizer_state = qr_optimizer.init(qr_params)
    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
    normalizer_params = running_statistics.init_state(obs_shape)
    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        qr_optimizer_state=qr_optimizer_state,
        qr_params=qr_params,
        target_qr_params=qr_params,
        qc_optimizer_state=qc_optimizer_state,
        qc_params=qc_params,
        target_qc_params=qc_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
        penalizer_params=penalizer_params,
    )  #  type: ignore
    return training_state


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    get_experience_fn: CollectDataFn = collect_single_step,
    learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-4,
    cost_critic_learning_rate: float = 1e-4,
    alpha_learning_rate: float = 3e-4,
    init_alpha: float | None = None,
    min_alpha: float = 0.0,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    num_critic_updates_per_actor_update: int = 1,
    deterministic_eval: bool = False,
    network_factory: sac_networks.NetworkFactory[
        sac_networks.SafeSACNetworks
    ] = sac_networks.make_sac_networks,
    n_critics: int = 2,
    n_heads: int = 1,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    safe: bool = False,
    safety_budget: float = float("inf"),
    penalizer: Penalizer | None = None,
    penalizer_params: Params | None = None,
    reward_q_transform: QTransformation = SACBase(),
    cost_q_transform: QTransformation = SACCost(),
    use_bro: bool = True,
    normalize_budget: bool = True,
    reset_on_eval: bool = True,
    store_buffer: bool = False,
    use_rae: bool = False,
):
    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )
    episodic_safety_budget = safety_budget
    if safety_discounting != 1.0 and normalize_budget:
        safety_budget = (
            (safety_budget / episode_length)
            / (1.0 - safety_discounting)
            * action_repeat
        )
    logging.info(f"Episode safety budget: {safety_budget}")
    if max_replay_size is None:
        max_replay_size = num_timesteps
    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * num_envs
    env_steps_per_experience_call = env_steps_per_actor_step
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_experience_call = -(-min_replay_size // num_envs)
    if get_experience_fn != collect_single_step:
        # Using episodic or hardware (which is episodic)
        env_steps_per_experience_call *= episode_length
        num_prefill_experience_call = -(-num_prefill_experience_call // episode_length)
    num_prefill_env_steps = num_prefill_experience_call * env_steps_per_experience_call
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_experience_call)
    )
    env = environment
    if wrap_env_fn is not None:
        env = wrap_env_fn(env)
    rng = jax.random.PRNGKey(seed)
    obs_size = env.observation_size
    action_size = env.action_size
    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        safe=safe,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    alpha_optimizer = optax.adam(learning_rate=alpha_learning_rate)
    make_optimizer = lambda lr, grad_clip_norm: optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(learning_rate=lr),
    )
    policy_optimizer = make_optimizer(learning_rate, 1.0)
    qr_optimizer = make_optimizer(critic_learning_rate, 1.0)
    qc_optimizer = make_optimizer(cost_critic_learning_rate, 1.0) if safe else None
    if isinstance(obs_size, Mapping):
        dummy_obs = {k: jnp.zeros(v) for k, v in obs_size.items()}
    else:
        dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    extras = {
        "state_extras": {
            "truncation": jnp.zeros(()),
        },
        "policy_extras": {},
    }
    if safe:
        extras["state_extras"]["cost"] = jnp.zeros(())  # type: ignore
    if isinstance(cost_q_transform, UCBCost):
        extras["state_extras"]["disagreement"] = jnp.zeros(())  # type: ignore
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras=extras,
    )
    dummy_transition = float16(dummy_transition)
    global_key, local_key = jax.random.split(rng)
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        qr_optimizer=qr_optimizer,
        qc_optimizer=qc_optimizer,
        penalizer_params=penalizer_params,
    )
    del global_key
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        penalizer_params = type(training_state.penalizer_params)(**params[2])
        training_state = training_state.replace(  # type: ignore
            normalizer_params=params[0],
            policy_params=params[1],
            penalizer_params=penalizer_params,
            qr_params=params[3],
            qc_params=params[4],
        )
        if len(params) >= 6 and use_rae:
            logging.info("Restoring replay buffer state")
            buffer_state = params[5]
            buffer_state = replay_buffers.ReplayBufferState(**buffer_state)
            replay_buffer = RAEReplayBuffer(
                max_replay_size=max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=batch_size * grad_updates_per_step,
                offline_data_state=buffer_state,
            )
    if not restore_checkpoint_path or not use_rae:
        replay_buffer = replay_buffers.UniformSamplingQueue(
            max_replay_size=max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=batch_size * grad_updates_per_step,
        )
    buffer_state = replay_buffer.init(rb_key)
    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        discounting=discounting,
        safety_discounting=safety_discounting,
        action_size=action_size,
        init_alpha=init_alpha,
        use_bro=use_bro,
    )
    alpha_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=None
        )
    )
    critic_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qr_optimizer, pmap_axis_name=None
        )
    )
    if safe:
        cost_critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qc_optimizer, pmap_axis_name=None
        )
    actor_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=None, has_aux=True
        )
    )
    extra_fields = ("truncation",)
    if safe:
        extra_fields += ("cost",)  # type: ignore
    if isinstance(cost_q_transform, UCBCost):
        extra_fields += ("disagreement",)  # type: ignore

    training_step = make_training_step(
        env,
        make_policy,
        replay_buffer,
        alpha_update,
        critic_update,
        cost_critic_update,
        actor_update,
        safe,
        min_alpha,
        reward_q_transform,
        cost_q_transform,
        penalizer,
        grad_updates_per_step,
        extra_fields,
        get_experience_fn,
        env_steps_per_experience_call,
        safety_budget,
        tau,
        num_critic_updates_per_actor_update,
    )

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience_fn(
                env,
                make_policy,
                training_state.policy_params,
                training_state.normalizer_params,
                replay_buffer,
                env_state,
                buffer_state,
                key,
                extra_fields,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_experience_call,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_experience_call,
        )[0]

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime  # type: ignore
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            env_steps_per_experience_call * num_training_steps_per_epoch
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            buffer_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Training state init
    # Env init
    env_keys = jax.random.split(env_key, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(env_keys)

    # Replay buffer init

    if not eval_env:
        eval_env = environment
    evaluator = ConstraintsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        budget=episodic_safety_budget,
        num_episodes=num_eval_episodes,
    )

    # Run initial eval
    metrics = {}
    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.policy_params),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    replay_size = jnp.sum(replay_buffer.size(buffer_state))
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)
        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_key
        )
        if reset_on_eval:
            reset_keys = jax.random.split(epoch_key, num_envs)
            env_state = reset_fn(reset_keys)
        current_step = int(training_state.env_steps)

        # Eval and logging
        if checkpoint_logdir:
            # Save current policy.
            params = (
                training_state.normalizer_params,
                training_state.policy_params,
                training_state.penalizer_params,
                training_state.qr_params,
                training_state.qc_params,
            )
            if store_buffer:
                params += (buffer_state,)
            dummy_ckpt_config = config_dict.ConfigDict()
            checkpoint.save(checkpoint_logdir, current_step, params, dummy_ckpt_config)

        # Run evals.
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.policy_params),
            training_metrics,
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps
    params = (
        training_state.normalizer_params,
        training_state.policy_params,
        training_state.penalizer_params,
        training_state.qr_params,
        training_state.qc_params,
    )
    if store_buffer:
        params += (buffer_state,)
    logging.info("total steps: %s", total_steps)
    return make_policy, params, metrics
