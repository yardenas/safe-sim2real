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

import functools
import time
from typing import Callable, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.envs.wrappers.training import VmapWrapper
from brax.training import replay_buffers, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.sac import checkpoint
from brax.training.types import PRNGKey, Transition
from etils import epath
from jax._src.lax import slicing
from ml_collections import config_dict
from orbax import checkpoint as ocp

from ss2r.algorithms.mb_ppo import Metrics, TrainingState, model_env
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.mb_ppo import model_train_step as mb_model_training_step
from ss2r.algorithms.mb_ppo import networks as mb_ppo_networks
from ss2r.algorithms.mb_ppo import pre_train_model as ptm
from ss2r.algorithms.mb_ppo import training_step as mb_ppo_training_step
from ss2r.algorithms.sac.data import collect_single_step
from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.evaluation import ConstraintsEvaluator


def _index_array(i, aval, x):
    return slicing.index_in_dim(x, i, keepdims=False)


def _scan(f, init, xs, length=None, reverse=False, unroll=1, *, use_lax=True):
    if use_lax:
        return jax.lax.scan(f, init, xs, length=length, reverse=reverse, unroll=unroll)
    else:
        xs_flat, xs_tree = jax.tree_flatten(xs)
        carry = init
        ys = []
        maybe_reversed = reversed if reverse else lambda x: x
        for i in maybe_reversed(range(length)):
            xs_slice = [_index_array(i, jax._src.core.get_aval(x), x) for x in xs_flat]
            carry, y = f(carry, jax.tree_unflatten(xs_tree, xs_slice))
        ys.append(y)
        stack = lambda *ys: jax.numpy.stack(ys)
        stacked_y = jax.tree_map(stack, *maybe_reversed(ys))
        return carry, stacked_y


def _init_training_state(
    key,
    ppo_network,
    model_optimizer,
    policy_optimizer,
    value_optimizer,
    cost_value_optimizer,
    obs_size,
):
    key_policy, key_value, key_cost_value = jax.random.split(key, 3)
    init_model_ensemble = jax.vmap(ppo_network.model_network.init)
    # TODO (yarden): hadrocede 5
    model_keys = jax.random.split(key_policy, 5)
    init_params = mb_ppo_losses.MBPPOParams(
        model=init_model_ensemble(model_keys),
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=ppo_network.cost_value_network.init(key_cost_value),
    )  # type: ignore
    model_optimizer_state = model_optimizer.init(init_params.model)
    policy_optimizer_state = policy_optimizer.init(init_params.policy)
    value_optimizer_state = value_optimizer.init(init_params.value)
    cost_value_optimizer_state = cost_value_optimizer.init(init_params.cost_value)
    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
    normalizer_params = running_statistics.init_state(obs_shape)
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=(
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=normalizer_params,
        env_steps=0,
    )  # type: ignore
    return training_state, init_params


def train(
    environment: envs.Env,
    num_timesteps,
    episode_length: int,
    update_step_factory=mb_ppo_training_step.update_fn,
    model_update_step_factory=mb_model_training_step.update_fn,
    get_experience_fn: CollectDataFn = collect_single_step,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    actor_critic_learning_rate: float = 1e-4,
    model_learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 1024,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    model_updates_per_step: int = 1000,
    ppo_updates_per_step: int = 1,
    num_evals: int = 1,
    min_replay_size: int = 0,
    max_replay_size: int = 1000000,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        mb_ppo_networks.MBPPONetworks
    ] = mb_ppo_networks.make_mb_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    restore_checkpoint_path: Optional[str] = None,
    checkpoint_logdir: Optional[str] = None,
    safety_budget: float = float("inf"),
    safe: bool = False,
    use_bro: bool = True,
    n_ensemble: int = 5,
    ensemble_selection: str = "random",
    learn_std: bool = False,
    normalize_budget: bool = True,
    reset_on_eval: bool = True,
    pretrain_model: bool = False,
    pretrain_epochs: int = 5000,
    pretrain_num_samples: int = 1000000,
    store_buffer: bool = False,
):
    # Check if environment and buffer params are compatible.
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
        env_steps_per_experience_call *= episode_length // action_repeat
        num_prefill_experience_call = -(
            -num_prefill_experience_call // (episode_length * action_repeat)
        )
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
    num_evals_after_init = max(num_evals - 1, 1)
    # Random nuber generator keys
    rng = jax.random.PRNGKey(seed)
    env = environment
    obs_size = env.observation_size
    action_size = env.action_size
    normalize_fn = lambda x, y: x
    denormalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
        denormalize_fn = running_statistics.denormalize
    ppo_network = network_factory(
        obs_size,
        action_size,
        preprocess_observations_fn=normalize_fn,
        postprocess_observations_fn=denormalize_fn,
        use_bro=use_bro,
        n_ensemble=n_ensemble,
        learn_std=learn_std,
    )
    make_policy = mb_ppo_networks.make_inference_fn(ppo_network)
    model_optimizer = optax.adam(learning_rate=model_learning_rate)
    policy_optimizer = optax.adam(learning_rate=actor_critic_learning_rate)
    value_optimizer = optax.adam(learning_rate=actor_critic_learning_rate)
    cost_value_optimizer = optax.adam(learning_rate=actor_critic_learning_rate)
    if max_grad_norm is None:
        model_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=model_learning_rate),
        )
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=actor_critic_learning_rate),
        )
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=actor_critic_learning_rate),
        )
        cost_value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=actor_critic_learning_rate),
        )

    # Create replay buffer
    # We need to accurately capture the shape of observations from the environment
    if isinstance(obs_size, Mapping):
        dummy_obs = {k: jnp.zeros(v) for k, v in obs_size.items()}
    else:
        dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    extras = {
        "state_extras": {
            "truncation": jnp.zeros(()),
        },
        "policy_extras": {
            "log_prob": jnp.zeros(()),
            "raw_action": jnp.zeros((action_size,)),
        },
    }
    if safe:
        extras["state_extras"]["cost"] = jnp.zeros(())  # type: ignore

    # Create properly structured dummy transition
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=jnp.zeros(()),
        discount=jnp.zeros(()),
        next_observation=dummy_obs,
        extras=extras,
    )
    dummy_transition = float16(dummy_transition)
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        # Ensure we sample enough transitions to divide evenly into minibatches
        sample_batch_size=batch_size,
    )
    # Make losses
    model_loss, policy_loss, value_loss, cost_value_loss = mb_ppo_losses.make_losses(
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        safety_discounting=safety_discounting,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        normalize_fn=normalize_fn,
    )

    # Create the model-based planning environment
    def create_planning_env(model_params, normalizer_params, ensemble_selection):
        """Create a planning environment with correct batch size for model-based rollouts."""
        planning_env = model_env.create_model_env(
            model_network=ppo_network.model_network,
            model_params=model_params,
            normalizer_params=normalizer_params,
            ensemble_selection=ensemble_selection,
            safety_budget=safety_budget,
            observation_size=obs_size,
            action_size=action_size,
        )
        planning_env = VmapWrapper(planning_env)
        return planning_env

    # Creating the PPO update step
    training_step = update_step_factory(
        policy_loss,
        value_loss,
        cost_value_loss,
        policy_optimizer,
        value_optimizer,
        cost_value_optimizer,
        create_planning_env,  # Pass the factory function
        replay_buffer,
        unroll_length,
        num_minibatches,
        make_policy,
        num_updates_per_batch,
        safe,
        ensemble_selection,
    )

    # Creating the model training step
    model_training_step = model_update_step_factory(
        model_loss,
        model_optimizer,
        replay_buffer,
        learn_std=learn_std,
    )

    extra_fields = ("truncation",)
    if safe:
        extra_fields += ("cost",)  # type: ignore

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
            training_state.params.policy,
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
                training_state.params.policy,
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

        training_state, env_state, buffer_state, new_key = _scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_experience_call,
        )[0]
        new_key, model_key = jax.random.split(new_key, 2)
        # Learn model from sampled transitions
        (
            (training_state, buffer_state, _),
            _,
        ) = _scan(
            model_training_step,
            (training_state, buffer_state, model_key),
            (),
            length=pretrain_epochs,
        )
        return training_state, env_state, buffer_state, new_key

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        # Run experience collection (later online)

        def g(carry, unused_t):
            # Learn model from sampled transitions
            training_state, env_state, buffer_state, key = carry
            key, model_key, actor_critic_key = jax.random.split(key, 3)
            (
                (training_state, buffer_state, _),
                model_loss_metrics,
            ) = _scan(
                model_training_step,
                (training_state, buffer_state, model_key),
                (),
                length=1,
            )
            # Learn model with ppo on learned model (planning MDP)
            (
                (training_state, buffer_state, _),
                ppo_loss_metrics,
            ) = _scan(
                training_step,
                (training_state, buffer_state, actor_critic_key),
                (),
                length=1,
            )
            metrics = model_loss_metrics | ppo_loss_metrics
            return (training_state, env_state, buffer_state, key), metrics

        def f(carry, unused_t):
            training_state, env_state, buffer_state, key = carry
            training_state, env_state, buffer_state, training_key = run_experience_step(
                training_state, env_state, buffer_state, key
            )
            (training_state, env_state, buffer_state, key), metrics = _scan(
                g,
                (training_state, env_state, buffer_state, key),
                (),
                length=ppo_updates_per_step,
            )
            return (
                training_state,
                env_state,
                buffer_state,
                training_key,
            ), metrics

        (training_state, env_state, buffer_state, key), metrics = _scan(
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
        result = training_epoch(training_state, env_state, buffer_state, key)
        training_state, env_state, buffer_state, metrics = result
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

    global_key, local_key = jax.random.split(rng)
    training_state, init_params = _init_training_state(
        global_key,
        ppo_network,
        model_optimizer,
        policy_optimizer,
        value_optimizer,
        cost_value_optimizer,
        obs_size,
    )
    del global_key
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)
    # Initialize replay buffer state
    buffer_state = replay_buffer.init(rb_key)
    # Training state init
    # Env init
    env_keys = jax.random.split(env_key, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(env_keys)
    if (
        restore_checkpoint_path is not None
        and epath.Path(restore_checkpoint_path).exists()
    ):
        logging.info("restoring from checkpoint %s", restore_checkpoint_path)
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        target = training_state.normalizer_params, init_params
        (normalizer_params, init_params) = orbax_checkpointer.restore(
            restore_checkpoint_path, item=target
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            params=init_params,
        )  # type: ignore

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

    # Pretrain the model
    if pretrain_model:
        (model_params, normalizer_params) = ptm.pre_train_model(
            params=training_state.params.model,  # type: ignore
            model_network=ppo_network.model_network,
            normalizer_params=training_state.normalizer_params,
            normalizer_fn=normalize_fn,
            model_optimizer=model_optimizer,
            optimizer_state=training_state.optimizer_state[0],  # type: ignore
            env=env,
            model_loss=model_loss,
            num_samples=pretrain_num_samples,
            batch_size=batch_size,
            epochs=pretrain_epochs,
        )
        training_state = training_state.replace(  # type: ignore
            params=training_state.params.replace(model=model_params),  # type: ignore
            normalizer_params=normalizer_params,
            optimizer_state=(model_optimizer.init(model_params),)
            + training_state.optimizer_state[1:],  # type: ignore
        )
        pretrained_model = training_state.params.model  # type: ignore
    else:
        pretrained_model = None
    # Run initial eval
    metrics = {}
    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.params.policy),
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
                training_state.params.policy,
                training_state.params.value,
                training_state.params.cost_value,
                training_state.params.model,
                pretrained_model,
            )
            if store_buffer:
                params += (buffer_state,)  # type: ignore
            dummy_ckpt_config = config_dict.ConfigDict()
            checkpoint.save(checkpoint_logdir, current_step, params, dummy_ckpt_config)

        # Run evals.
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.params.policy),
            training_metrics,
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps
    params = (
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
        training_state.params.cost_value,
        training_state.params.model,
        pretrained_model,  # type: ignore
    )
    if store_buffer:
        params += (buffer_state,)  # type: ignore
    logging.info("total steps: %s", total_steps)
    return make_policy, params, metrics
