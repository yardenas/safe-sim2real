import functools
from typing import Sequence, Tuple

import jax
from brax import envs
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.replay_buffers import ReplayBuffer
from brax.training.types import Params, PRNGKey

from ss2r.algorithms.sac.types import CollectDataFn, ReplayBufferState, float16
from ss2r.rl.types import MakePolicyFn


# JIT compile the actor step function
@jax.jit
def ppo_actor_step(
    env,
    env_state,
    make_policy_fn,    # This is already a configured policy function that takes (obs, key)
    policy_params,     # Not used directly here - already configured in make_policy_fn
    normalizer_params, # Not used directly here - already configured in make_policy_fn
    key,               # Overall PRNG key for this step
    extra_fields,
    deterministic=False, # Controls if the created policy should be deterministic in its behavior
):
    """Actor step function that properly handles PPO's policy interface."""
    
    # This policy_fn_for_actor_step will be called by acting.actor_step as:
    # policy_fn_for_actor_step(observations, key_from_actor_step_for_policy)
    def policy_fn_for_actor_step(observations, key_for_policy_call):
        # make_policy_fn is already configured with parameters and just needs (obs, key)
        return make_policy_fn(observations, key_for_policy_call)

    # acting.actor_step will use the 'key' to derive a key for policy_fn_for_actor_step and for env.step
    return acting.actor_step(env, env_state, policy_fn_for_actor_step, key, extra_fields)


def make_ppo_collection_fn(unroll_fn) -> CollectDataFn:
    """Creates a collection function for PPO policies."""
    
    # JIT compile the collection function
    @jax.jit
    def collect_data(
        env: envs.Env,
        make_policy_fn: MakePolicyFn, # This is the policy template
        params: Params, # These are the policy_params (model weights)
        normalizer_params: running_statistics.RunningStatisticsState,
        replay_buffer: ReplayBuffer,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
        extra_fields: Sequence[str] = ("truncation",),
        # Add deterministic flag if it needs to be passed down to unroll_fn
        # deterministic: bool = False, # Example
    ) -> Tuple[
        running_statistics.RunningStatisticsState,
        envs.State,
        ReplayBufferState,
    ]:
        # Call the unroll function (ppo_actor_step)
        # It expects make_policy_fn (template), policy_params (weights), normalizer_params
        env_state, transitions = unroll_fn(
            env,
            env_state,
            make_policy_fn,    # The policy template
            params,            # Policy parameters (model weights)
            normalizer_params, # Normalizer parameters
            key,
            extra_fields=extra_fields,
            # deterministic=deterministic # Pass if ppo_actor_step needs it directly from here
                                        # or if it's already part of its own args
        )
        
        # Reshape transitions if needed
        if transitions.reward.ndim == 2:
            transitions = jax.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), transitions
            )
            
        # Update normalizer with new observations
        new_normalizer_params = running_statistics.update(
            normalizer_params, transitions.observation
        )
        
        # Add transitions to replay buffer
        new_buffer_state = replay_buffer.insert(buffer_state, float16(transitions))
        
        return new_normalizer_params, env_state, new_buffer_state

    return collect_data


# Create the collect_single_step function for PPO
collect_ppo_single_step = make_ppo_collection_fn(ppo_actor_step)