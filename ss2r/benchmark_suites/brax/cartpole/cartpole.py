import os
from typing import Any

import jax
import jax.numpy as jnp
from brax import base
from brax.envs import Env, Wrapper, register_environment
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from ss2r.benchmark_suites import rewards


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        gain = sys.actuator.gain.copy()
        gain_sample = jax.random.uniform(rng, minval=cfg.gain[0], maxval=cfg.gain[1])
        gain = gain.at[0].add(gain_sample)
        rng, _ = jax.random.split(rng)
        gear = sys.actuator.gear.copy()[0]
        gear_sample = (
            jax.random.uniform(rng, minval=cfg.gear[0], maxval=cfg.gear[1]) + gear
        )
        return gain, gear_sample

    actuator_gain, actuator_gear = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "actuator.gain": 0,
            "actuator.gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "actuator.gain": actuator_gain,
            "actuator.gear": actuator_gear[:, None],
        }
    )
    samples = jnp.stack(
        [
            actuator_gain[:, 0],
            actuator_gear,
        ],
        axis=-1,
    )
    return sys, in_axes, samples


class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env, slider_position_bound: float):
        assert isinstance(env, Cartpole)
        super().__init__(env)
        self.slider_position_bound = slider_position_bound

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        slider_pos = self.env.cart_position(nstate.pipeline_state)
        cost = (jnp.abs(slider_pos) >= self.slider_position_bound).astype(jnp.float32)
        nstate.info["cost"] = cost
        return nstate


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


for safe in [True, False]:
    name = ["cartpole"]
    safe_str = "safe" if safe else ""
    if safe:
        name.append("safe")

        def make(**kwargs):
            slider_position_bound = kwargs.pop("slider_position_bound", 0.25)
            return ConstraintWrapper(
                Cartpole(**kwargs),
                slider_position_bound,
            )
    else:

        def make(**kwargs):
            kwargs.pop("slider_position_bound", None)
            return Cartpole(**kwargs)

    name_str = "_".join(name)
    register_environment(name_str, make)
