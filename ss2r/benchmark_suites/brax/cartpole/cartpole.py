from typing import Any
import jax

import jax.numpy as jnp
from brax import base
from brax.envs import register_environment
from brax.envs.base import PipelineEnv, State
from ss2r.algorithms.jax.state_sampler import StateSampler
from ss2r.benchmark_suites.brax import rewards


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        cpole = (
            jax.random.normal(rng) * cfg.scale + sys.link.inertia.mass[-1] + cfg.shift
        )
        mass = sys.link.inertia.mass.at[-1].set(cpole)
        return mass, cpole

    mass, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"link.inertia.mass": 0})
    sys = sys.tree_replace({"link.inertia.mass": mass})
    return sys, in_axes, samples[:, None]


def sample_state(state_sampler: StateSampler):
    pass


@register_environment("cartpole_swingup")
class CartpoleSwingup(PipelineEnv):
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        q = self.sys.init_q + jax.random.normal(rng1, (self.sys.q_size(),))
        q = q.at[1].add(jnp.pi)
        # FIXME (yarden): in dm control the qpos has dim 4.
        qd = jax.random.normal(rng2, (self.sys.qd_size())) * 0.01
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward, done = jnp.zeros(2)
        metrics: dict[str, Any] = {}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        obs = self._get_obs(pipeline_state)
        done = False
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=self._reward(pipeline_state),
            done=done,
        )

    def cart_position(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.q[0]

    def pole_angle_components(self, pipeline_state: base.State) -> jax.Array:
        return jnp.cos(pipeline_state.q[1]), jnp.sin(pipeline_state.q[1])

    def bounded_position(self, pipeline_state: base.State) -> jax.Array:
        return jnp.hstack(
            (
                self.cart_position(pipeline_state),
                *self.pole_angle_components(pipeline_state),
            )
        )

    def _reward(self, pipeline_state: base.State) -> jax.Array:
        cart_in_bounds = rewards.tolerance(
            self.cart_position(pipeline_state), (-0.25, 0.25)
        )
        angle_in_bounds = rewards.tolerance(
            self.pole_angle_components(pipeline_state)[0], (0.995, 1.0)
        )
        return cart_in_bounds * angle_in_bounds

    @property
    def action_size(self):
        return 1

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        # FIXME (yarden): qd here is not like in dm_control
        return jnp.concatenate(
            [self.bounded_position(pipeline_state), pipeline_state.qd]
        )
