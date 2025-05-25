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

"""Model-based Proximal Policy Optimization training."""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Dict, Sequence
from brax.training import gradients

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from brax import envs
from brax.training import types, replay_buffers
from brax.training.acme import running_statistics, specs
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.penalizers import Penalizer
from ss2r.algorithms.mb_ppo.model import make_world_model_ensemble
from ss2r.algorithms.mb_ppo.losses import make_losses
from ss2r.algorithms.ppo import networks as ppo_networks
from ss2r.algorithms.ppo import losses as ppo_losses
from ss2r.algorithms.mb_ppo.data import collect_ppo_single_step  # Add this import
from ss2r.algorithms.sac.types import (
    CollectDataFn,
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
    float32,
)
from ss2r.rl.evaluation import ConstraintsEvaluator


@flax.struct.dataclass
class ModelState:
    """Contains training state for the world model."""
    model_params: Params
    model_optimizer_state: optax.OptState
    gradient_steps: jnp.ndarray


@flax.struct.dataclass
class PPOState:
    """Contains training state for the PPO agent."""
    policy_params: Params
    value_params: Params
    cost_value_params: Params
    policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    cost_value_optimizer_state: optax.OptState


@flax.struct.dataclass
class TrainingState:
    """Contains combined training state for the learner."""
    model_state: ModelState
    ppo_state: PPOState
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray
    penalizer_params: Params


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    action_size: int,
    ppo_network: ppo_networks.SafePPONetworks,
    world_model,
    model_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    cost_value_optimizer: optax.GradientTransformation,
    penalizer_params: Params = None,
    n_ensemble: int = 5,
    use_bro: bool = True,
    predict_std: bool = False,
) -> TrainingState:
    """Initialize the training state."""
    # Initialize keys
    key_policy, key_value, key_model = jax.random.split(key, 3)
    
    # Initialize world model parameters
    model_params = world_model.init(key_model)
    model_optimizer_state = model_optimizer.init(model_params)
    
    # Initialize policy and value function parameters
    policy_params = ppo_network.policy_network.init(key_policy)
    value_params = ppo_network.value_network.init(key_value)
    cost_value_params = ppo_network.cost_value_network.init(key_value)
    
    policy_optimizer_state = policy_optimizer.init(policy_params)
    value_optimizer_state = value_optimizer.init(value_params)
    cost_value_optimizer_state = cost_value_optimizer.init(cost_value_params)
    
    # Initialize normalizer state
    if isinstance(obs_size, Mapping):
        obs_shape = {
            k: specs.Array(v, jnp.dtype("float32")) for k, v in obs_size.items()
        }
    else:
        obs_shape = specs.Array((obs_size,), jnp.dtype("float32"))
    normalizer_params = running_statistics.init_state(obs_shape)
    
    # Create Model state
    model_state = ModelState(
        model_params=model_params,
        model_optimizer_state=model_optimizer_state,
        gradient_steps=jnp.zeros(()),
    )
    
    # Create PPO state
    ppo_state = PPOState(
        policy_params=policy_params,
        value_params=value_params,
        cost_value_params=cost_value_params,
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        cost_value_optimizer_state=cost_value_optimizer_state,
    )
    
    # Create combined training state
    training_state = TrainingState(
        model_state=model_state,
        ppo_state=ppo_state,
        normalizer_params=normalizer_params,
        env_steps=jnp.zeros(()),
        penalizer_params=penalizer_params,
    )
    
    return training_state


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    get_experience_fn: CollectDataFn = collect_ppo_single_step,
    learning_rate: float = 1e-4,
    model_learning_rate: float = 1e-4,
    critic_learning_rate: float = 1e-4,
    cost_critic_learning_rate: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    model_batch_size: int = 256,
    num_model_updates_per_step: int = 4,
    unroll_length: int = 10,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    entropy_cost: float = 1e-4,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    safety_gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    model_rollout_length: int = 5,
    model_rollouts_per_step: int = 100,
    n_ensemble: int = 5,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    learn_std: bool = False,
    deterministic_eval: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    eval_env: Optional[envs.Env] = None,
    safety_budget: float = float("inf"),
    penalizer: Penalizer | None = None,
    penalizer_params: Params | None = None,
    use_bro: bool = True,
    normalize_budget: bool = True,
):
    """Train a model-based PPO agent."""
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
        
    # Environment setup
    env_steps_per_actor_step = action_repeat * num_envs
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    
    # Calculate the number of training steps per epoch
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )
    
    env = environment
    if wrap_env_fn is not None:
        env = wrap_env_fn(env)
        
    rng = jax.random.PRNGKey(seed)
    obs_size = env.observation_size
    action_size = env.action_size
    
    # Create normalizing function
    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize
    
    # Create world model ensemble
    world_model = make_world_model_ensemble(
        obs_size=obs_size,
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes,
        n_ensemble=n_ensemble,
        use_bro=use_bro,
        predict_std=learn_std,
    )
    
    # Create PPO networks
    ppo_network = ppo_networks.make_ppo_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        policy_hidden_layer_sizes=policy_hidden_layer_sizes,
        value_hidden_layer_sizes=value_hidden_layer_sizes,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    # Create optimizers
    model_optimizer = optax.adam(learning_rate=model_learning_rate)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    value_optimizer = optax.adam(learning_rate=critic_learning_rate)
    cost_value_optimizer = optax.adam(learning_rate=cost_critic_learning_rate)
    
    # Create losses
    model_loss_fn = make_losses(world_model)
    policy_loss_fn, value_loss_fn, cost_value_loss_fn = ppo_losses.make_losses(
        ppo_network=ppo_network,
        clipping_epsilon=clipping_epsilon,
        entropy_cost=entropy_cost,
        reward_scaling=reward_scaling,
        discounting=discounting,
        gae_lambda=gae_lambda,
        normalize_advantage=normalize_advantage,
        cost_scaling=cost_scaling,
        safety_budget=safety_budget,
        safety_discounting=safety_discounting,
        safety_gae_lambda=safety_gae_lambda,
        use_saute=False,
        use_disagreement=False,
    )
    
    # Create gradient update functions
    model_update_fn = gradients.gradient_update_fn(
        model_loss_fn, model_optimizer, pmap_axis_name=None, has_aux=True
    )
    policy_update_fn = gradients.gradient_update_fn(
        policy_loss_fn, policy_optimizer, pmap_axis_name=None, has_aux=True
    )
    value_update_fn = gradients.gradient_update_fn(
        value_loss_fn, value_optimizer, pmap_axis_name=None, has_aux=True
    )
    cost_value_update_fn = gradients.gradient_update_fn(
        cost_value_loss_fn, cost_value_optimizer, pmap_axis_name=None, has_aux=True
    )
    
    # Setup replay buffer for real experience
    extra_fields = ("truncation",)
    if penalizer is not None:
        extra_fields += ("cost",)
    
    # Create dummy transition for replay buffer initialization
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
    if penalizer is not None:
        extras["state_extras"]["cost"] = jnp.zeros(())
        extras["state_extras"]["cumulative_cost"] = jnp.zeros(())
        
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
        sample_batch_size=model_batch_size,
    )
    
    global_key, local_key = jax.random.split(rng)
    
    # Initialize training state
    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        action_size=action_size,
        ppo_network=ppo_network,
        world_model=world_model,
        model_optimizer=model_optimizer,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        cost_value_optimizer=cost_value_optimizer,
        penalizer_params=penalizer_params,
        n_ensemble=n_ensemble,
        use_bro=use_bro,
        predict_std=learn_std,
    )
    
    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Initialize environment
    env_keys = jax.random.split(env_key, num_envs)
    reset_fn = jax.jit(env.reset)
    env_state = reset_fn(env_keys)

    # Initialize replay buffer
    buffer_state = replay_buffer.init(rb_key)

    if not eval_env:
        eval_env = environment
    
    # Setup evaluator
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
    
    training_walltime = 0.0

    # Run initial evaluation
    metrics = {}
    if num_evals > 1:
        metrics = evaluator.run_evaluation(
            (
                training_state.normalizer_params,
                training_state.ppo_state.policy_params,
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Create functions for the training loop

    @jax.jit
    def run_experience_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        """Collects experience from real environment."""
        experience_key, training_key = jax.random.split(key)
        
        # Create policy function from the current parameters
        policy_fn = make_policy((
            training_state.normalizer_params,
            training_state.ppo_state.policy_params,
            training_state.ppo_state.value_params,
        ))
        
        # Collect experience from the environment
        normalizer_params, env_state, buffer_state = get_experience_fn(
            env,
            policy_fn,
            training_state.ppo_state.policy_params,
            training_state.normalizer_params,
            replay_buffer,
            env_state,
            buffer_state,
            experience_key,
            extra_fields,
        )
        
        # Update training state with new normalizer params and step count
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )
        return training_state, env_state, buffer_state, training_key

    @jax.jit
    def update_model(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, Metrics]:
        """Updates the world model parameters using sampled experience."""
        model_key, key = jax.random.split(key)
        
        # Sample transitions from replay buffer
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = float32(transitions)

        # Repeat sampling and updating for multiple model updates
        def update_step(carry, _):
            state, key = carry
            key, update_key = jax.random.split(key)
            
            # Get model state from training state
            model_params = state.model_state.model_params
            model_optimizer_state = state.model_state.model_optimizer_state
            
            # Update model using sampled transitions
            (loss_val, mse), model_params, model_optimizer_state = model_update_fn(
                model_params,
                state.normalizer_params,
                transitions,
                update_key,
                learn_std,
                optimizer_state=model_optimizer_state,
            )
            
            # Create new model state
            new_model_state = ModelState(
                model_params=model_params,
                model_optimizer_state=model_optimizer_state,
                gradient_steps=state.model_state.gradient_steps + 1,
            )
            
            # Update the training state with the new model state
            new_state = state.replace(model_state=new_model_state)
            metrics = {"model_loss": loss_val, "model_mse": mse}
            
            return (new_state, key), metrics
        
        # Run multiple model updates
        (training_state, _), metrics = jax.lax.scan(
            update_step,
            (training_state, model_key),
            None,
            length=num_model_updates_per_step,
        )
        
        # Compute mean metrics
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, metrics

    @jax.jit
    def generate_model_rollouts(
        training_state: TrainingState,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[types.Transition, PRNGKey]:
        """Generate rollouts using the model for policy learning."""
        rollout_key, key = jax.random.split(key)
        
        # Sample initial states from the buffer
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        initial_obs = transitions.observation
        
        # Convert to float32 to avoid dtype mismatch
        initial_obs = initial_obs.astype(jnp.float32)
        
        # Define model rollout function
        def rollout_step(carry, _):
            obs, rollout_key = carry
            rollout_key, action_key, next_key = jax.random.split(rollout_key, 3)
            
            # Get policy action
            policy_fn = make_policy((
                training_state.normalizer_params,
                training_state.ppo_state.policy_params,
                training_state.ppo_state.value_params,
            ))
            action, policy_extras = policy_fn(obs, action_key)
            
            # Predict next state using the model
            pred_next_obs, _ = world_model.apply(
                training_state.normalizer_params,  # Add this parameter
                training_state.model_state.model_params,
                next_key,  # Add the key parameter
                obs, 
                action,
            )

            # Convert to float32 and handle ensemble
            pred_next_obs = pred_next_obs.astype(jnp.float32)
            if len(pred_next_obs.shape) > 2:
                pred_next_obs = jnp.mean(pred_next_obs, axis=1)
                
            # Get value function estimates
            value = ppo_network.value_network.apply(
                training_state.normalizer_params,
                training_state.ppo_state.value_params,
                obs,
            )
            cost_value = ppo_network.cost_value_network.apply(
                training_state.normalizer_params,
                training_state.ppo_state.cost_value_params,
                obs,
            )
            
            # Approximate reward and cost as 
            value_next = ppo_network.value_network.apply(
                training_state.normalizer_params,
                training_state.ppo_state.value_params,
                pred_next_obs,
            )

            cost_value_next = ppo_network.cost_value_network.apply(
                training_state.normalizer_params,
                training_state.ppo_state.cost_value_params,
                pred_next_obs,
            )

            # Calculate reward and cost
            reward = value_next - value
            cost = cost_value_next - cost_value 

             # Create extras - fix the shapes
            extras = {
                "state_extras": {
                    "truncation": jnp.zeros_like(reward),
                    "value": value.squeeze(),
                    "cost_value": cost_value.squeeze(),
                },
                "policy_extras": policy_extras,
            }
            
            if penalizer is not None:
                extras["state_extras"]["cost"] = cost
                extras["state_extras"]["cumulative_cost"] = cost
                
            # Create transition
            transition = types.Transition(
                observation=obs,
                action=action,
                reward=reward,
                discount=jnp.ones_like(reward) * discounting,
                next_observation=pred_next_obs,
                extras=extras,
            )
            
            return (pred_next_obs, next_key), transition
        
        # Fixed: Generate multiple independent rollouts with correct signature
        def generate_single_rollout(carry, _):
            key = carry  # carry is the key
            key, rollout_key = jax.random.split(key)
            
            # Generate a complete rollout starting from sampled initial states
            (_, _), rollout = jax.lax.scan(
                rollout_step,
                (initial_obs, rollout_key),
                None,
                length=model_rollout_length,
            )
            
            return key, rollout  # Return updated key and the rollout
        
        # Generate multiple rollouts
        final_key, all_rollouts = jax.lax.scan(
            generate_single_rollout,
            key,  # Initial carry (the key)
            None, # xs (not used, hence None)
            length=model_rollouts_per_step,
        )
        
        # Reshape the rollouts for PPO training
        # all_rollouts shape: (num_rollouts, rollout_length, batch_size, ...)
        # Flatten the first two dimensions to create a large batch
        batch_rollouts = jax.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), 
            all_rollouts
        )
        
        return batch_rollouts, final_key

    @jax.jit
    def update_policy(
        training_state: TrainingState,
        rollouts: types.Transition,
        key: PRNGKey,
    ) -> Tuple[TrainingState, Metrics]:
        """Updates policy parameters using PPO on model rollouts."""
        ppo_key, key = jax.random.split(key)
        
        # Create policy update function
        def policy_update_step(carry, _):
            state, update_key = carry
            update_key, key_policy = jax.random.split(update_key)
            
            # Update policy
            (_, policy_aux), policy_params, policy_optimizer_state = policy_update_fn(
                state.ppo_state.policy_params,
                state.ppo_state.value_params,
                state.ppo_state.cost_value_params,
                state.normalizer_params,
                rollouts,
                penalizer,
                state.penalizer_params,
                key_policy,
                optimizer_state=state.ppo_state.policy_optimizer_state,
            )
            
            # Update value function
            (_, value_aux), value_params, value_optimizer_state = value_update_fn(
                state.ppo_state.value_params,
                state.normalizer_params,
                rollouts,
                optimizer_state=state.ppo_state.value_optimizer_state,
            )
            
            # Update cost value function if using a penalizer
            if penalizer is not None:
                (_, cost_value_aux), cost_value_params, cost_value_optimizer_state = cost_value_update_fn(
                    state.ppo_state.cost_value_params,
                    state.normalizer_params,
                    rollouts,
                    optimizer_state=state.ppo_state.cost_value_optimizer_state,
                    params=state.ppo_state.cost_value_params,
                )
                
                # Update penalizer parameters
                penalizer_aux, new_penalizer_params = penalizer.update(
                    policy_aux["normalized_constraint_estimate"], 
                    state.penalizer_params
                )
                policy_aux.update(penalizer_aux)
                policy_aux.update(cost_value_aux)
            else:
                cost_value_params = state.ppo_state.cost_value_params
                cost_value_optimizer_state = state.ppo_state.cost_value_optimizer_state
                new_penalizer_params = state.penalizer_params
            
            # Create new PPO state
            new_ppo_state = PPOState(
                policy_params=policy_params,
                value_params=value_params,
                cost_value_params=cost_value_params,
                policy_optimizer_state=policy_optimizer_state,
                value_optimizer_state=value_optimizer_state,
                cost_value_optimizer_state=cost_value_optimizer_state,
            )
            
            # Update the training state with the new PPO state and penalizer params
            new_state = state.replace(
                ppo_state=new_ppo_state,
                penalizer_params=new_penalizer_params,
            )
            
            metrics = {
                **policy_aux,
                **value_aux,
            }
            
            return (new_state, update_key), metrics
        
        # Run policy updates
        (training_state, _), metrics = jax.lax.scan(
            policy_update_step,
            (training_state, ppo_key),
            None,
            length=num_updates_per_batch,
        )
        
        # Compute mean metrics
        metrics = jax.tree_map(jnp.mean, metrics)
        return training_state, metrics

    @jax.jit
    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        """Prefills the replay buffer with initial experience."""
        def f(carry, _):
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            
            # Create policy function from the current parameters
            policy_fn = make_policy((
                training_state.normalizer_params,
                training_state.ppo_state.policy_params,
                training_state.ppo_state.value_params,
            ))
            
            # Collect experience
            normalizer_params, env_state, buffer_state = get_experience_fn(
                env,
                policy_fn,
                training_state.ppo_state.policy_params,
                training_state.normalizer_params,
                replay_buffer,
                env_state,
                buffer_state,
                key,
                extra_fields,
            )
            
            # Update training state
            new_training_state = training_state.replace(
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            None,
            length=num_prefill_actor_steps,
        )[0]

    @jax.jit
    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        """Runs one training epoch of model-based PPO."""
        def f(carry, _):
            ts, es, bs, k = carry
            
            # Split random keys
            k, exp_key, model_key, rollout_key, policy_key = jax.random.split(k, 5)
            
            # Step 1: Collect experience from the real environment
            ts, es, bs, _ = run_experience_step(ts, es, bs, exp_key)
            
            # Step 2: Update the world model using real data
            ts, model_metrics = update_model(ts, bs, model_key)
            
            # Step 3: Generate synthetic rollouts using the model
            rollouts, _ = generate_model_rollouts(ts, bs, rollout_key)
            
            # Step 4: Update the policy using PPO on the rollouts
            ts, policy_metrics = update_policy(ts, rollouts, policy_key)
            
            # Combine metrics
            metrics = {**model_metrics, **policy_metrics}
            if hasattr(es, 'metrics'):
                metrics.update(es.metrics)
            
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            None,
            length=num_training_steps_per_epoch,
        )
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        """Wraps training_epoch with timing info."""
        nonlocal training_walltime
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
        
        return training_state, env_state, buffer_state, metrics

    # Prefill the replay buffer
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    replay_size = jnp.sum(replay_buffer.size(buffer_state))
    logging.info(f"Replay size after prefill: {replay_size}")
    assert replay_size >= min_replay_size
    
    training_walltime = time.time() - t

    # Main training loop
    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info(f"Step {current_step}")
        
        # Run training epoch
        epoch_key, local_key = jax.random.split(local_key)
        
        (
            training_state,
            env_state,
            buffer_state,
            training_metrics,
        ) = training_epoch_with_timing(
            training_state, env_state, buffer_state, epoch_key
        )
        
        # Update step counter
        current_step = int(training_state.env_steps)

        # Run evaluation
        metrics = evaluator.run_evaluation(
            (training_state.normalizer_params, training_state.ppo_state.policy_params),
            training_metrics,
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps
    
    params = (
        training_state.normalizer_params,
        training_state.ppo_state.policy_params,
        training_state.ppo_state.value_params,
        training_state.model_state.model_params,
    )
    
    logging.info(f"Total steps: {total_steps}")
    return make_policy, params, metrics
