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
from brax.training.agents.ppo.train import (
    _random_translate_pixels as batch_random_translate_pixels,
)
from brax.training.agents.sac import checkpoint
from brax.training.types import Params, PRNGKey
from ml_collections import config_dict

import ss2r.algorithms.sac.losses as sac_losses
import ss2r.algorithms.sac.networks as sac_networks
from ss2r.algorithms.penalizers import Penalizer
from ss2r.algorithms.sac import gradients
from ss2r.algorithms.sac.data import collect_single_step
from ss2r.algorithms.sac.q_transforms import QTransformation, SACBase, SACCost, UCBCost
from ss2r.algorithms.sac.rae import RAEReplayBufferState
from ss2r.algorithms.sac.types import (
    CollectDataFn,
    Metrics,
    ReplayBufferState,
    TrainingState,
    Transition,
    float16,
    float32,
)
from ss2r.rl.evaluation import ConstraintsEvaluator
from ss2r.rl.utils import (
    dequantize_images,
    quantize_images,
    remove_pixels,
    restore_state,
)


def _random_translate_pixels(x, rng):
    x = jax.tree_map(lambda x: x[:, None], x)
    y = batch_random_translate_pixels(x, rng)
    return jax.tree_map(lambda y: y[:, 0], y)


def update_lr_schedule_count(opt_state, new_count):
    """Return a copy of opt_state with updated learning rate schedule count."""
    # Deepcopy to avoid in-place mutation
    # Update the learning rate scheduler count
    if opt_state is None:
        return
    state = opt_state[-1]
    if "learning_rate" in state.hyperparams_states:
        old_hyperparams_state = state.hyperparams_states["learning_rate"]
        updated_schedule_state = old_hyperparams_state._replace(count=new_count)
        state = state._replace(
            hyperparams_states={
                **state.hyperparams_states,
                "learning_rate": updated_schedule_state,
            }
        )
        opt_state = opt_state[0], state
    else:
        raise KeyError("No 'learning_rate' key found in hyperparams_states")

    return opt_state


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    sac_network: sac_networks.SafeSACNetworks,
    init_alpha: float,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    qr_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation | None,
    penalizer_params: Params | None,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_qr, key_qc = jax.random.split(key, 3)
    log_alpha = jnp.asarray(jnp.log(init_alpha), dtype=jnp.float32)
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
    normalizer_params = running_statistics.init_state(remove_pixels(obs_shape))
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
    init_alpha: float = 1.0,
    min_alpha: float = 0.0,
    target_entropy: float | None = None,
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
    replay_buffer_factory: Callable[
        [int, Any, PRNGKey, int], replay_buffers.ReplayBuffer
    ] = replay_buffers.UniformSamplingQueue,
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
    schedule_lr: bool = False,
    init_lr: float = 0.0,
    actor_burnin: float = 0.0,
    actor_wait: float = 0.0,
    critic_burnin: float = 0.0,
    entropy_bonus: bool = True,
    augment_pixels: bool = False,
    load_buffer: bool = False,
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
    if isinstance(obs_size, Mapping):
        for key, value in obs_size.items():
            if key.startswith("pixels/") and len(value) > 3 and value[0] == 1:
                value = value[1:]
                obs_size[key] = value  # type: ignore
    action_size = env.action_size
    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        safe=safe and penalizer is not None,
        use_bro=use_bro,
        n_critics=n_critics,
        n_heads=n_heads,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)
    alpha_optimizer = optax.adam(learning_rate=alpha_learning_rate)
    make_optimizer = lambda lr, grad_clip_norm, grad_steps, wait=0: optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=optax.schedules.linear_schedule(
                init_lr if schedule_lr else lr, lr, grad_steps, wait
            )
        ),
    )
    num_grad_steps = num_training_steps_per_epoch * grad_updates_per_step * num_evals
    policy_optimizer = make_optimizer(
        learning_rate,
        1.0,
        int(num_grad_steps * actor_burnin),
        int(num_grad_steps * actor_wait),
    )
    qr_optimizer = make_optimizer(
        critic_learning_rate, 1.0, int(num_grad_steps * critic_burnin)
    )
    qc_optimizer = (
        make_optimizer(
            cost_critic_learning_rate, 1.0, int(num_grad_steps * critic_burnin)
        )
        if safe and penalizer is not None
        else None
    )
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
    dummy_transition = dummy_transition._replace(
        observation=quantize_images(dummy_transition.observation),
        next_observation=quantize_images(dummy_transition.next_observation),
    )
    global_key, local_key = jax.random.split(rng)
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        sac_network=sac_network,
        init_alpha=init_alpha,
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
        policy_optimizer_state = update_lr_schedule_count(
            restore_state(params[6], training_state.policy_optimizer_state), 0
        )
        alpha_optimizer_state = restore_state(
            params[7], training_state.alpha_optimizer_state
        )
        qr_optimizer_state = update_lr_schedule_count(
            restore_state(params[8], training_state.qr_optimizer_state), 0
        )
        if qc_optimizer is None:
            qc_optimizer_state = None
        else:
            qc_optimizer_state = update_lr_schedule_count(
                restore_state(params[9], training_state.qc_optimizer_state), 0
            )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=params[0],
            policy_params=params[1],
            penalizer_params=restore_state(params[2], training_state.penalizer_params),
            qr_params=params[3],
            target_qr_params=params[3],
            qc_params=params[4],
            target_qc_params=params[4],
            alpha_params=params[5],
            policy_optimizer_state=policy_optimizer_state,
            alpha_optimizer_state=alpha_optimizer_state,
            qr_optimizer_state=qr_optimizer_state,
            qc_optimizer_state=qc_optimizer_state,
        )
    replay_buffer = replay_buffer_factory(  # type: ignore
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size,
    )
    buffer_state = replay_buffer.init(rb_key)
    if load_buffer:
        data = Transition(**params[-1].pop("data"))
        buffer_state = buffer_state.replace(**params[-1], data=data)  # type: ignore
    alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        discounting=discounting,
        safety_discounting=safety_discounting,
        action_size=action_size,
        use_bro=use_bro,
        target_entropy=target_entropy,
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

    def sgd_step(
        carry: Tuple[TrainingState, ReplayBufferState, PRNGKey, int], unused_t
    ) -> Tuple[Tuple[TrainingState, ReplayBufferState, PRNGKey, int], Metrics]:
        training_state, buffer_state, key, count = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)
        new_buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = float32(transitions)
        if augment_pixels:
            key, key_obs, key_next_obs = jax.random.split(key, 3)
            observations = _random_translate_pixels(
                dequantize_images(transitions.observation), key_obs
            )
            next_observations = _random_translate_pixels(
                dequantize_images(transitions.next_observation), key_next_obs
            )
            transitions = transitions._replace(
                observation=observations, next_observation=next_observations
            )
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        if entropy_bonus:
            alpha = jnp.exp(training_state.alpha_params) + min_alpha
        else:
            alpha = 0.0
        critic_loss, qr_params, qr_optimizer_state = critic_update(
            training_state.qr_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qr_params,
            alpha,
            transitions,
            key_critic,
            reward_q_transform,
            optimizer_state=training_state.qr_optimizer_state,
            params=training_state.qr_params,
        )
        if safe and penalizer is not None:
            cost_critic_loss, qc_params, qc_optimizer_state = cost_critic_update(
                training_state.qc_params,
                training_state.policy_params,
                training_state.normalizer_params,
                training_state.target_qc_params,
                alpha,
                transitions,
                key_critic,
                cost_q_transform,
                True,
                optimizer_state=training_state.qc_optimizer_state,
                params=training_state.qc_params,
            )
            cost_metrics = {
                "cost_critic_loss": cost_critic_loss,
            }
        else:
            cost_metrics = {}
            qc_params = None
            qc_optimizer_state = None
        if augment_pixels:
            encoder_params = qr_params["params"]["SharedEncoder"]
            policy_params = training_state.policy_params.copy()
            policy_params["params"]["SharedEncoder"] = encoder_params
        else:
            policy_params = training_state.policy_params
        # TODO (yarden): try to make it faster with cond later
        (actor_loss, aux), new_policy_params, new_policy_optimizer_state = actor_update(
            policy_params,
            training_state.normalizer_params,
            training_state.qr_params,
            training_state.qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            penalizer,
            training_state.penalizer_params,
            optimizer_state=training_state.policy_optimizer_state,
            params=policy_params,
        )
        should_update_actor = count % num_critic_updates_per_actor_update == 0
        update_if_needed = lambda x, y: jnp.where(should_update_actor, x, y)
        policy_params = jax.tree_map(
            update_if_needed, new_policy_params, training_state.policy_params
        )
        policy_optimizer_state = jax.tree_map(
            update_if_needed,
            new_policy_optimizer_state,
            training_state.policy_optimizer_state,
        )
        polyak = lambda target, new: jax.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_target_qr_params = polyak(training_state.target_qr_params, qr_params)
        if safe:
            new_target_qc_params = polyak(training_state.target_qc_params, qc_params)
        else:
            new_target_qc_params = None
        if aux:
            new_penalizer_params = aux.pop("penalizer_params")
            additional_metrics = {
                **aux,
            }
        else:
            new_penalizer_params = training_state.penalizer_params
            additional_metrics = {}

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
            **cost_metrics,
            **additional_metrics,
            "policy_lr": training_state.policy_optimizer_state[-1].hyperparams[
                "learning_rate"
            ],
            "critic_lr": training_state.qr_optimizer_state[-1].hyperparams[
                "learning_rate"
            ],
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
            penalizer_params=new_penalizer_params,
        )  # type: ignore
        return (new_training_state, new_buffer_state, key, count + 1), metrics

    def run_experience_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        """Runs the non-jittable experience collection step."""
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience_fn(
            env,
            make_policy,
            training_state.policy_params,
            training_state.normalizer_params,
            replay_buffer,
            env_state,
            buffer_state,
            experience_key,
            extra_fields,
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_experience_call,
        )
        return training_state, env_state, buffer_state, training_key

    def training_step_jitted(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        training_key: PRNGKey,
    ) -> Tuple[TrainingState, ReplayBufferState, Metrics]:
        """Runs the jittable training step after experience collection."""
        (training_state, buffer_state, *_), metrics = jax.lax.scan(
            sgd_step,
            (training_state, buffer_state, training_key, 0),
            (),
            length=grad_updates_per_step,
        )
        return training_state, buffer_state, metrics

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        """Splits training into experience collection and a jitted training step."""
        training_state, env_state, buffer_state, training_key = run_experience_step(
            training_state, env_state, buffer_state, key
        )
        training_state, buffer_state, training_metrics = training_step_jitted(
            training_state, buffer_state, training_key
        )
        training_metrics |= env_state.metrics
        training_metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        if isinstance(buffer_state, RAEReplayBufferState):
            training_metrics["rae_mixing"] = replay_buffer.mix(buffer_state.step)
        return training_state, env_state, buffer_state, training_metrics

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
                training_state.alpha_params,
                training_state.policy_optimizer_state,
                training_state.alpha_optimizer_state,
                training_state.qr_optimizer_state,
                training_state.qc_optimizer_state,
            )
            if store_buffer:
                if isinstance(buffer_state, RAEReplayBufferState):
                    params += (buffer_state.online_state,)
                else:
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
        training_state.alpha_params,
        training_state.policy_optimizer_state,
        training_state.alpha_optimizer_state,
        training_state.qr_optimizer_state,
        training_state.qc_optimizer_state,
    )
    if store_buffer:
        if isinstance(buffer_state, RAEReplayBufferState):
            params += (buffer_state.online_state,)
        else:
            params += (buffer_state,)
    logging.info("total steps: %s", total_steps)
    return make_policy, params, metrics
