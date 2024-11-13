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
from typing import Any, Callable, Optional, Tuple, TypeAlias, Union

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import base, envs
from brax.io import model
from brax.training import acting, gradients, pmap, replay_buffers, types
from brax.training.acme import running_statistics, specs
from brax.training.types import Params, PRNGKey
from brax.v1 import envs as envs_v1

import ss2r.algorithms.sac.losses as sac_losses
import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.sac.robustness import SACCost
from ss2r.algorithms.sac.wrappers import DomainRandomizationParams, StatePropagation
from ss2r.rl.evaluation import ConstraintsEvaluator

Metrics: TypeAlias = types.Metrics
Transition: TypeAlias = types.Transition
InferenceParams: TypeAlias = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState: TypeAlias = Any

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    policy_optimizer_state: optax.OptState
    policy_params: Params
    qr_optimizer_state: optax.OptState
    qc_optimizer_state: optax.OptState | None
    qr_params: Params
    qc_params: Params | None
    target_qr_params: Params
    target_qc_params: Params | None
    gradient_steps: jnp.ndarray
    env_steps: jnp.ndarray
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    normalizer_params: running_statistics.RunningStatisticsState
    lagrangian_params: sac_losses.LagrangianParams


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_networks.SafeSACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    qr_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation | None,
    lagrange_multiplier: float,
    penalty_multiplier: float,
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
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype("float32"))
    )

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
        lagrangian_params=sac_losses.LagrangianParams(
            lagrange_multiplier, penalty_multiplier
        ),
    )  #  type: ignore
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps,
    episode_length: int,
    wrap_env: bool = True,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_trajectories_per_env: int = 1,
    cost_penalty: float | None = None,
    propagation: str | None = None,
    learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-4,
    cost_critic_learning_rate: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
    deterministic_eval: bool = False,
    network_factory: sac_networks.DomainRandomizationNetworkFactory[
        sac_networks.SafeSACNetworks
    ] = sac_networks.make_sac_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System, jax.Array]]
    ] = None,
    privileged: bool = False,
    safe: bool = False,
    safety_budget: float = float("inf"),
    lagrange_multiplier: float = 1e-9,
    penalty_multiplier: float = 1.0,
    penalty_multiplier_factor: float = 1.0,
    cost_q_transform: str | None = None,
    cvar_confidence: float = 0.95,
):
    """SAC training."""
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()
    logging.info(
        "local_device_count: %s; total_device_count: %s",
        local_devices_to_use,
        device_count,
    )

    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )

    safety_budget = (safety_budget / episode_length) / (1.0 - safety_discounting)
    logging.info(f"Episode safety budget: {safety_budget}")
    if max_replay_size is None:
        max_replay_size = num_timesteps
    if propagation == "standard":
        propagation = None
    factor = 1 if propagation is not None else num_envs
    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = action_repeat * factor * num_trajectories_per_env
    # equals to ceil(min_replay_size / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-min_replay_size // (factor * num_trajectories_per_env))
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    rng = jax.random.PRNGKey(seed)
    if wrap_env:
        if isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            wrap_for_training = envs_v1.wrappers.wrap_for_training

        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, num_envs // jax.process_count() // local_devices_to_use
                ),
            )
            sys, in_axes, domain_parameters = v_randomization_fn(env.sys)
            vf_randomization_fn = lambda *_: (sys, in_axes)
        else:
            vf_randomization_fn = None
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=vf_randomization_fn,
        )
    if privileged:
        env = DomainRandomizationParams(env)
        domain_parameters = env.domain_parameters
    else:
        domain_parameters = None
    if propagation is not None:
        env = StatePropagation(env)

    obs_size = env.observation_size
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    domain_randomization_size = (
        domain_parameters.shape[-1] if domain_parameters is not None else 0
    )
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        domain_randomization_size=domain_randomization_size,
        safe=safe,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)

    policy_optimizer = optax.adam(learning_rate=learning_rate)
    qr_optimizer = optax.adam(learning_rate=critic_learning_rate)
    qc_optimizer = optax.adam(learning_rate=cost_critic_learning_rate) if safe else None

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    extras = {
        "state_extras": {
            "truncation": 0.0,
        },
        "policy_extras": {},
    }
    if domain_parameters is not None:
        extras["state_extras"]["domain_parameters"] = domain_parameters[0]  # type: ignore
    if safe:
        if propagation is not None:
            extras["state_extras"]["state_propagation"] = {  # type: ignore
                "next_obs": jnp.tile(dummy_obs, (num_envs,) + (1,) * dummy_obs.ndim),
                "rng": rng,
            }
        extras["state_extras"]["cost"] = 0.0  # type: ignore

    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras=extras,
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
    )

    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        safety_discounting=safety_discounting,
        action_size=action_size,
    )
    alpha_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )
    critic_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qr_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    )
    if safe:
        cost_critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            critic_loss, qc_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
        )
    actor_update = (
        gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
            actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )
    )

    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_cost_critic, key_actor = jax.random.split(
            key, 5
        )

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, qr_params, qr_optimizer_state = critic_update(
            training_state.qr_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qr_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.qr_optimizer_state,
        )
        if safe:
            cost_critic_loss, qc_params, qc_optimizer_state = cost_critic_update(
                training_state.qc_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_qc_params,
                alpha,
                transitions,
                key_critic,
                True,
                SACCost(),
                optimizer_state=training_state.qc_optimizer_state,
            )
            cost_metrics = {
                "cost_critic_loss": cost_critic_loss,
            }
        else:
            cost_metrics = {}
            qc_params = None
            qc_optimizer_state = None
        (actor_loss, aux), policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.qr_params,
            training_state.qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            training_state.lagrangian_params,
            optimizer_state=training_state.policy_optimizer_state,
        )
        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)
        if safe:
            new_target_qc_params = polyak(training_state.target_qc_params, qc_params)
        else:
            new_target_qc_params = None
        if aux:
            additional_metrics = {
                **aux,
            }
            new_lagrangian_params = training_state.lagrangian_params
        else:
            new_lagrangian_params = training_state.lagrangian_params
            additional_metrics = {}

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            **cost_metrics,
            **additional_metrics,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            qr_optimizer_state=qr_optimizer_state,
            qc_optimizer_state=qc_optimizer_state,
            qr_params=qr_params,
            qc_params=qc_params,
            target_qr_params=new_target_qr_params,
            target_qc_params=new_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
            lagrangian_params=new_lagrangian_params,
        )  # type: ignore
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: Union[envs.State, envs_v1.State],
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        Union[envs.State, envs_v1.State],
        ReplayBufferState,
    ]:
        policy = make_policy((normalizer_params, policy_params))
        extra_fields = ("truncation",)
        if domain_parameters is not None:
            extra_fields += ("domain_parameters",)  # type: ignore
        if propagation is not None:
            extra_fields += ("state_propagation",)  # type: ignore
        if safe:
            extra_fields += ("cost",)  # type: ignore
        step = lambda state: acting.actor_step(
            env, state, policy, key, extra_fields=extra_fields
        )
        step = jax.vmap(step)
        env_state, transitions = step(env_state)
        normalizer_params = running_statistics.update(
            normalizer_params, transitions.observation, pmap_axis_name=_PMAP_AXIS_NAME
        )
        if transitions.observation.ndim == 3:
            transitions = jax.tree_util.tree_map(lambda x: x[0], transitions)
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics
    ]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

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
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

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

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

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
            env_steps_per_actor_step * num_training_steps_per_epoch
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

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        qr_optimizer=qr_optimizer,
        qc_optimizer=qc_optimizer,
        lagrange_multiplier=lagrange_multiplier,
        penalty_multiplier=penalty_multiplier,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(
        env_key, num_trajectories_per_env * num_envs // jax.process_count()
    )
    env_keys = jnp.reshape(
        env_keys,
        (local_devices_to_use, num_trajectories_per_env, -1) + env_keys.shape[1:],
    )
    env_state = jax.pmap(jax.vmap(env.reset))(env_keys)

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    if not eval_env:
        eval_env = environment
    if wrap_env:
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
            )
            vf_randomization_fn = lambda sys: v_randomization_fn(sys)[:-1]  # type: ignore
        else:
            vf_randomization_fn = None
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=vf_randomization_fn,
        )

    evaluator = ConstraintsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    replay_size = (
        jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    logging.info("replay size after prefill %s", replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info("step %s", current_step)

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_keys
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Save current policy.
                params = _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                )
                path = f"{checkpoint_logdir}_sac_{current_step}.pkl"
                model.save_params(path, params)

            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps

    params = _unpmap((training_state.normalizer_params, training_state.policy_params))

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return make_policy, params, metrics
