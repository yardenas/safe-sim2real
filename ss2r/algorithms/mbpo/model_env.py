import jax
import jax.numpy as jnp
from brax import envs
from brax.envs import base

from ss2r.algorithms.sac.types import float32


class ModelBasedEnv(envs.Env):
    """Environment wrapper that uses a learned model for predictions."""

    def __init__(
        self,
        transitions,
        observation_size,
        action_size,
        model_network,
        model_params,
        qc_network,
        qc_params,
        normalizer_params,
        ensemble_selection="mean",  # "random", "mean", or "pessimistic"
        safety_budget=float("inf"),
        cost_discount=1.0,
        scaling_fn=lambda x: x,  # Function to scale costs
    ):
        super().__init__()
        self.model_network = model_network
        self.model_params = model_params
        self.qc_network = qc_network
        self.qc_params = qc_params
        self.normalizer_params = normalizer_params
        self.ensemble_selection = ensemble_selection
        self.safety_budget = safety_budget
        self._observation_size = observation_size
        self._action_size = action_size
        self.transitions = transitions
        self.cost_discount = cost_discount
        self.scaling_fn = scaling_fn

    def reset(self, rng: jax.Array) -> base.State:
        sample_key, model_key = jax.random.split(rng)
        indcs = jax.random.randint(sample_key, (), 0, self.transitions.reward.shape[0])
        transitions = float32(
            jax.tree_util.tree_map(lambda x: x[indcs], self.transitions)
        )
        state = envs.State(
            pipeline_state=None,
            obs=transitions.observation,
            reward=transitions.reward,
            done=jnp.zeros_like(transitions.reward),
            info={
                "truncation": jnp.zeros_like(transitions.reward),
                "cost": transitions.extras["state_extras"].get(
                    "cost", jnp.zeros_like(transitions.reward)
                ),
                "key": model_key,
            },
        )
        return state

    def step(self, state: base.State, action: jax.Array) -> base.State:
        """Step using the learned model."""
        # Predict next state, reward, and cost using the model
        key = state.info["key"]
        sample_key, key = jax.random.split(key)
        next_obs, reward, cost = _propagate_ensemble(
            self.model_network.apply,
            self.normalizer_params,
            self.model_params,
            state.obs,
            action,
            self.ensemble_selection,
            sample_key,
        )
        state.info["key"] = key
        done = jnp.zeros_like(reward, dtype=jnp.float32)
        truncation = jnp.zeros_like(reward, dtype=jnp.float32)
        state.info["cost"] = cost
        state.info["truncation"] = truncation
        if self.qc_network is not None:
            prev_cumulative_cost = state.obs["cumulative_cost"][0]
            curr_discount = state.obs["curr_discount"][0] * self.cost_discount
            expected_cost_for_traj = (
                self.scaling_fn(prev_cumulative_cost)
                + self.qc_network.apply(
                    self.normalizer_params,
                    self.qc_params,
                    state.obs,
                    action,
                ).mean(axis=-1)
                * curr_discount
            )
            done = jnp.where(
                expected_cost_for_traj > self.safety_budget,
                jnp.ones_like(done),
                done,
            )

            def reset_states(self, done, state, next_obs):
                """Reset the state if done."""
                key, reset_keys = jax.random.split(state.info["key"])
                state.info["key"] = key
                next_obs["cumulative_cost"] = (
                    state.obs["cumulative_cost"] + cost * curr_discount
                )
                next_obs["curr_discount"] = curr_discount
                obs = {
                    k: jnp.where(done, self.reset(reset_keys).obs[k], next_obs[k])
                    for k in next_obs.keys()
                }
                return state, obs

            state, next_obs = reset_states(self, done, state, next_obs)

        state = state.replace(
            obs=next_obs,
            reward=reward,
            done=done,
            info=state.info,
        )
        return state

    def observation_size(self) -> int:
        """Return the size of the observation space."""
        return self._observation_size

    def action_size(self) -> int:
        """Return the size of the action space."""
        return self._action_size

    def backend(self) -> str:
        """Return the backend used by the environment."""
        return "model_based"


def create_model_env(
    transitions,
    model_network,
    model_params,
    qc_network,
    qc_params,
    observation_size,
    action_size,
    normalizer_params,
    ensemble_selection="random",
    safety_budget=float("inf"),
    cost_discount=1.0,
    scaling_fn=lambda x: x,  # Function to scale costs
) -> ModelBasedEnv:
    """Factory function to create a model-based environment."""
    return ModelBasedEnv(
        transitions=transitions,
        model_network=model_network,
        observation_size=observation_size,
        action_size=action_size,
        model_params=model_params,
        qc_network=qc_network,
        qc_params=qc_params,
        normalizer_params=normalizer_params,
        ensemble_selection=ensemble_selection,
        safety_budget=safety_budget,
        cost_discount=cost_discount,
        scaling_fn=scaling_fn,
    )


def _propagate_ensemble(
    pred_fn,
    normalizer_params,
    model_params,
    obs,
    action,
    ensemble_selection,
    key,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Propagate the ensemble predictions based on the selection method."""
    # Calculate the nominal predictions
    if ensemble_selection == "nominal":
        # Get the average model parameters
        avg_model_params = jax.tree_util.tree_map(
            lambda p: jnp.mean(p, axis=0), model_params
        )
        next_obs, reward, cost = pred_fn(
            normalizer_params, avg_model_params, obs, action
        )
    elif ensemble_selection == "random":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        # Randomly select one of the ensemble predictions
        idx = jax.random.randint(key, (1,), 0, reward_pred.shape[0])[0]
        if isinstance(next_obs_pred, dict):
            next_obs = {k: v[idx] for k, v in next_obs_pred.items()}
        else:
            next_obs = next_obs_pred[idx]
        reward = reward_pred[idx]
        cost = cost_pred[idx]
    elif ensemble_selection == "mean":
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward_pred, cost_pred = vmap_pred_fn(
            normalizer_params, model_params, obs, action
        )
        if isinstance(next_obs_pred, dict):
            next_obs = {k: jnp.mean(v, axis=0) for k, v in next_obs_pred.items()}
        else:
            next_obs = jnp.mean(next_obs_pred, axis=0)
        reward = jnp.mean(reward_pred, axis=0)
        cost = jnp.mean(cost_pred, axis=0)
    else:
        raise ValueError(f"Unknown ensemble selection: {ensemble_selection}")
    return next_obs, reward, cost
