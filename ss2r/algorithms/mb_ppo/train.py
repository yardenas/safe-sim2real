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

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from brax import envs
from brax.training import pmap, types, replay_buffers
from brax.training.acme import running_statistics, specs
from brax.training.types import PRNGKey, Transition
from brax.v1 import envs as envs_v1
from etils import epath
from orbax import checkpoint as ocp

from ss2r.algorithms.mb_ppo import _PMAP_AXIS_NAME, Metrics, TrainingState
from ss2r.algorithms.mb_ppo import losses as mb_ppo_losses
from ss2r.algorithms.mb_ppo import networks as mb_ppo_networks
from ss2r.algorithms.mb_ppo import training_step as mb_ppo_training_step
from ss2r.algorithms.mb_ppo import model_train_step as mb_model_training_step
from ss2r.algorithms.ppo.wrappers import TrackOnlineCosts
from ss2r.algorithms.sac.types import ReplayBufferState, CollectDataFn, float16
from ss2r.algorithms.sac.data import collect_single_step
from ss2r.rl.evaluation import ConstraintsEvaluator
from ss2r.algorithms.mb_ppo import model_env


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    update_step_factory=mb_ppo_training_step.update_fn,
    model_update_step_factory=mb_model_training_step.update_fn,
    get_experience_fn: CollectDataFn = collect_single_step,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    num_experience_steps_per_epoch: int = 10,
    min_replay_size: int = 0,
    max_replay_size: int = 1000000,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    safety_gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        mb_ppo_networks.MBPPONetworks
    ] = mb_ppo_networks.make_mb_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    restore_checkpoint_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,  # Add this parameter
    safety_budget: float = float("inf"),
    safe: bool = False,
    normalize_budget: bool = True,
    learn_std: bool = False,
):
    # Check if environment and buffer params are compatible.
    if min_replay_size >= num_timesteps:
        raise ValueError(
            "No training will happen because min_replay_size >= num_timesteps"
        )

    # Safety budget
    original_safety_budget = safety_budget
    if normalize_budget:
        safety_budget = (safety_budget / episode_length) / (1.0 - safety_discounting)
    xt = time.time()

    # Devices and process info.
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # Training step and evals
    env_steps_per_actor_step = action_repeat * num_envs * num_experience_steps_per_epoch
    num_prefill_actor_steps = -(-min_replay_size // num_envs)

    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_steps_per_actor_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    # Random nuber generator keys
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key, rb_key = jax.random.split(local_key, 4)

    key_policy, key_value = jax.random.split(global_key, 2)
    del global_key
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs,
        (local_devices_to_use, -1) + key_envs.shape[1:],
    )

    # Environment and evaluation setup
    assert num_envs % device_count == 0
    env = environment
    env = TrackOnlineCosts(env)
    reset_fn = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    action_size = env.action_size
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    # Create the model based PPO networks and optimizers
    ppo_network = network_factory(
        obs_shape, action_size, preprocess_observations_fn=normalize
    )
    make_policy = mb_ppo_networks.make_inference_fn(ppo_network)
    model_optimizer = optax.adam(learning_rate=learning_rate)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    value_optimizer = optax.adam(learning_rate=learning_rate)
    cost_value_optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is None:
        model_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        cost_value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    # Create replay buffer
    # We need to accurately capture the shape of observations from the environment
    sample_obs = jax.tree_util.tree_map(lambda x: x[0][0], env_state.obs)
    dummy_obs = sample_obs  # Use actual obs sample instead of zeros
    dummy_action = jnp.zeros((action_size,))

    # Debug the exact fields actually being used in data collection
    if safe:
        # For safe environments, the following fields should be included
        extras = {
            "truncation": jnp.zeros(()),
            "cost": jnp.zeros(()),
            "cumulative_cost": jnp.zeros(()),
            "episode_cost": jnp.zeros(()),  # This additional field might be needed
        }
    else:
        extras = {
            "truncation": jnp.zeros(()),
        }

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

    # Print detailed debug info
    logging.info(f"Replay buffer initialization:")
    logging.info(f"- Observation shape: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, dummy_obs)}")
    logging.info(f"- Extras fields: {list(extras.keys())}")
    logging.info(f"- Complete dummy transition structure: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, dummy_transition)}")

    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size,
        dummy_data_sample=dummy_transition,
        # Ensure we sample enough transitions to divide evenly into minibatches
        sample_batch_size=batch_size * num_minibatches,  # This matches the minibatch structure
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
        safety_gae_lambda=safety_gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        safety_budget=safety_budget,
    )

    # Create the model-based planning environment
    def create_planning_env(model_params, normalizer_params):
        """Create a planning environment with correct batch size for model-based rollouts."""
        # Set the planning batch size
        planning_batch_size = batch_size * num_minibatches
        
        # Extract the correct environment name by traversing the wrapper chain
        current_env = environment
        while hasattr(current_env, 'env'):
            current_env = current_env.env
        
        # Get the environment name and ensure it's lowercase for Brax compatibility
        if hasattr(current_env, 'name'):
            env_name = current_env.name.lower()  # Ensure lowercase for Brax
        else:
            env_name = current_env.__class__.__name__.lower()
        
        # Check for safe variant if using safe mode
        from brax import envs as brax_envs
        available_envs = list(brax_envs._envs.keys())
        if safe and f"{env_name}_safe" in available_envs:
            env_name = f"{env_name}_safe"
        
        logging.info(f"Creating planning environment with name '{env_name}' and batch size {planning_batch_size}")
        
        try:
            # Create a fresh environment instance with the correct batch size
            planning_base_env = brax_envs.create(env_name, batch_size=planning_batch_size)
            planning_base_env = TrackOnlineCosts(planning_base_env)
            
            # Wrap with model-based environment
            planning_env = model_env.create_model_env(
                env=planning_base_env,
                model_network=ppo_network.model_network,
                model_params=model_params,
                normalizer_params=normalizer_params,
                ensemble_selection="mean",
                safety_budget=safety_budget,
            )
            
            return planning_env
            
        except KeyError as e:
            logging.error(f"Failed to create environment '{env_name}'. Available: {available_envs}")
            # Try using default cartpole as fallback
            if "cartpole" in available_envs:
                logging.warning(f"Trying fallback to 'cartpole' environment")
                planning_base_env = brax_envs.create("cartpole", batch_size=planning_batch_size)
                planning_base_env = TrackOnlineCosts(planning_base_env)
                return model_env.create_model_env(
                    env=planning_base_env,
                    model_network=ppo_network.model_network,
                    model_params=model_params,
                    normalizer_params=normalizer_params,
                    ensemble_selection="mean",
                    safety_budget=safety_budget,
                )
            raise e

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
        batch_size,
        safe,
    )

    # Creating the model training step
    model_training_step = model_update_step_factory(
        model_loss,
        model_optimizer,
        replay_buffer,
        num_minibatches,
        num_updates_per_batch,
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
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )
        return training_state, env_state, buffer_state, training_key
    
    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:

        def experience_step_fn(carry, _):
            ts, es, bs, k = carry
            k, next_k = jax.random.split(k)
            ts, es, bs, _ = run_experience_step(ts, es, bs, k)
            return (ts, es, bs, next_k), None
        
        experience_carry = (training_state, env_state, buffer_state, key)

        (training_state, env_state, buffer_state, _), _ = jax.lax.scan(
            experience_step_fn,
            experience_carry,
            (),  
            length=num_experience_steps_per_epoch,
        )
        return (training_state, env_state, buffer_state, key), ()

    # Training epoch
    def training_epoch(
        training_state: TrainingState, env_state: envs.State, buffer_state:ReplayBufferState, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics, Metrics]:
        # Run experience collection (later online)

        def experience_step_fn(carry, _):
            ts, es, bs, k = carry
            k, next_k = jax.random.split(k)
            ts, es, bs, _ = run_experience_step(ts, es, bs, k)
            return (ts, es, bs, next_k), None
        
        experience_carry = (training_state, env_state, buffer_state, key)

        (training_state, env_state, buffer_state, _), _ = jax.lax.scan(
            experience_step_fn,
            experience_carry,
            (),  
            length=num_experience_steps_per_epoch,
        )

        # Learn model from sampled transitions
        (training_state, buffer_state, _), loss_metrics = jax.lax.scan(
            model_training_step,
            (training_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        model_loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)

        # Learn model with ppo on leaned model (planning MDP)
        (training_state, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, env_state, buffer_state, loss_metrics, model_loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, buffer_state:ReplayBufferState, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics, Metrics]:
        nonlocal training_walltime  # type: ignore
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, buffer_state, key)
        training_state, env_state, buffer_state, metrics, model_metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        model_metrics = jax.tree_util.tree_map(jnp.mean, model_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), model_metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_steps_per_actor_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
            **{f"model/{name}": value for name, value in model_metrics.items()},
        }
        return (
            training_state,
            env_state,
            buffer_state,
            metrics,
            model_metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = mb_ppo_losses.MBPPOParams(
        model=ppo_network.model_network.init(key_policy),
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=ppo_network.cost_value_network.init(key_value),
    )  # type: ignore
    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    model_optimizer_state = model_optimizer.init(init_params.model)
    policy_optimizer_state = policy_optimizer.init(init_params.policy)
    value_optimizer_state = value_optimizer.init(init_params.value)
    cost_value_optimizer_state = cost_value_optimizer.init(init_params.cost_value)
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=(
            model_optimizer_state,
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape),
        env_steps=0,
    )  # type: ignore

    # Initialize replay buffer state
    buffer_state = replay_buffer.init(rb_key)

    # Prefill replay buffer if needed
    if num_prefill_actor_steps > 0:
        logging.info(f"Prefilling replay buffer with {num_prefill_actor_steps * num_envs} steps...")
        prefill_start_time = time.time()
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, local_key
        )
        prefill_time = time.time() - prefill_start_time
        logging.info(f"Prefill complete. Time elapsed: {prefill_time:.2f}s")

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

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )
    buffer_state = jax.device_put_replicated(
    buffer_state, jax.local_devices()[:local_devices_to_use]
    )

    evaluator = ConstraintsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        budget=original_safety_budget,
        num_episodes=num_eval_episodes,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics: Metrics = {}
    model_trainig_metrics: Metrics = {}
    training_walltime = 0.0
    current_step = 0
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_metrics, model_trainig_metrics) = training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))
            key_env, tmp_key = jax.random.split(key_env)
            key_envs = jax.random.split(tmp_key, num_envs // process_count)
            key_envs = jnp.reshape(
                key_envs,
                (local_devices_to_use, -1) + key_envs.shape[1:],
            )
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                        training_state.params.value,
                    )
                ),
                training_metrics,
                model_trainig_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            params = _unpmap((training_state.normalizer_params, training_state.params))
            policy_params_fn(current_step, make_policy, params)

    total_steps = current_step
    print(f"Total steps: {total_steps}, expected: {num_timesteps}")
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
