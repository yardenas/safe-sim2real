import os
from typing import Any

import jax
import jax.numpy as jnp
from brax import base
from brax.envs import register_environment
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from ss2r.algorithms.state_sampler import StateSampler
from ss2r.benchmark_suites import rewards


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        # Hardcoding _POLE_MASS to avoid weird jax issues.
        pole_mass = jnp.asarray([1.0, 0.0])
        mask = jnp.asarray([0.0, 1.0])
        sample = jax.random.uniform(rng, minval=cfg.min, maxval=cfg.max) * mask
        sample = pole_mass + sample
        return sample

    samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"link.inertia.mass": 0})
    sys = sys.tree_replace({"link.inertia.mass": samples})
    return sys, in_axes, samples


def domain_randomization_length(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        offset = jax.random.uniform(rng, shape=(3,), minval=-0.1, maxval=0.1)
        pos = sys.link.transform.pos.at[0].set(offset)
        return pos

    pos = randomize(rng)
    sys_v = sys.tree_replace({"link.inertia.transform.pos": pos})
    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"link.inertia.transform.pos": 0})
    return sys_v, in_axes, pos


def domain_randomization_gear(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        sample = jax.random.uniform(rng, minval=cfg.min, maxval=cfg.max)
        return sample

    samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"actuator.gear": 0})
    sys = sys.tree_replace({"actuator.gear": samples})
    return sys, in_axes, samples


def sample_state(state_sampler: StateSampler):
    pass


class Cartpole(PipelineEnv):
    def __init__(self, backend="generalized", **kwargs):
        dir = os.path.dirname(__file__)
        path = os.path.join(dir, "cartpole.xml")
        sys = mjcf.load(path)
        self.sparse = kwargs.pop("sparse", False)
        self.swingup = kwargs.pop("swingup", False)
        super().__init__(sys=sys, backend=backend, n_frames=1, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        if self.swingup:
            q = self.sys.init_q + jax.random.normal(rng1, (self.sys.q_size(),)) * 0.01
            q = q.at[1].add(jnp.pi)
        else:
            q = self.sys.init_q
            q = q.at[0].set(jax.random.uniform(rng1, shape=(), minval=-1.0, maxval=1.0))
            q = q.at[1].set(
                jax.random.uniform(rng1, shape=(), minval=-0.034, maxval=0.034)
            )
        qd = jax.random.normal(rng2, (self.sys.qd_size(),)) * 0.01
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
        done = jnp.zeros_like(state.done)
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=self._reward(pipeline_state, action),
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

    def _reward(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self.sparse:
            cart_in_bounds = rewards.tolerance(
                self.cart_position(pipeline_state), (-0.25, 0.25)
            )
            angle_in_bounds = rewards.tolerance(
                self.pole_angle_components(pipeline_state)[0], (0.995, 1.0)
            )
            return cart_in_bounds * angle_in_bounds
        else:
            upright = (self.pole_angle_components(pipeline_state)[0] + 1) / 2
            centered = rewards.tolerance(self.cart_position(pipeline_state), margin=2)
            centered = (1 + centered) / 2
            small_control = rewards.tolerance(
                action, margin=1, value_at_margin=0, sigmoid="quadratic"
            )[0]
            small_control = (4 + small_control) / 5
            small_velocity = rewards.tolerance(pipeline_state.qd[1], margin=5).min()
            small_velocity = (1 + small_velocity) / 2
            return upright.mean() * small_control * small_velocity * centered

    @property
    def action_size(self):
        return 1

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe cartpole body position and velocities."""
        return jnp.concatenate(
            [self.bounded_position(pipeline_state), pipeline_state.qd]
        )


register_environment(
    "cartpole_swingup_sparse",
    lambda **kwargs: Cartpole(swingup=True, sparse=True, **kwargs),
)
register_environment(
    "cartpole_swingup", lambda **kwargs: Cartpole(swingup=True, **kwargs)
)
register_environment(
    "cartpole_balance", lambda **kwargs: Cartpole(swingup=False, **kwargs)
)
