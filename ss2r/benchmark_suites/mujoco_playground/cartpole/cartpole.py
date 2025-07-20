import functools

import jax
import jax.numpy as jnp
from brax.envs import Wrapper
from mujoco_playground import MjxEnv, State, dm_control_suite

from ss2r.benchmark_suites.rewards import tolerance

_POLE_ID = -1


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        pole_length_sample = jax.random.uniform(
            rng, minval=cfg.pole_length[0], maxval=cfg.pole_length[1]
        )
        length = 0.5 + pole_length_sample
        scale_factor = length / 0.5
        geom = sys.geom_size.copy()
        geom = geom.at[_POLE_ID, 1].set(length)
        mass = sys.body_mass.at[_POLE_ID].multiply(scale_factor)
        inertia = sys.body_inertia.at[_POLE_ID].multiply(scale_factor**3)
        mass_sample = jax.random.uniform(
            rng, minval=cfg.pole_mass[0], maxval=cfg.pole_mass[1]
        )
        scale = (sys.body_mass[_POLE_ID] + mass_sample) / sys.body_mass[_POLE_ID]
        mass = mass.at[_POLE_ID].multiply(scale)
        inertia = sys.body_inertia.at[_POLE_ID].multiply(scale)
        inertia_pos = sys.body_ipos.copy()
        inertia_pos = inertia_pos.at[_POLE_ID, -1].add(pole_length_sample / 2.0)
        gear = sys.actuator_gear.copy()
        gear_sample = jax.random.uniform(rng, minval=cfg.gear[0], maxval=cfg.gear[1])
        gear = gear.at[0, 0].add(gear_sample)
        return (
            inertia_pos,
            mass,
            inertia,
            geom,
            gear,
            pole_length_sample,
            mass_sample,
            gear_sample,
        )

    (
        inertia_pos,
        mass,
        inertia,
        geom,
        gear,
        length_sample,
        mass_sample,
        gear_sample,
    ) = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_mass": 0,
            "body_inertia": 0,
            "body_ipos": 0,
            "geom_size": 0,
            "actuator_gear": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_mass": mass,
            "body_inertia": inertia,
            "body_ipos": inertia_pos,
            "geom_size": geom,
            "actuator_gear": gear,
        }
    )
    return sys, in_axes, jnp.hstack([length_sample, mass_sample, gear_sample])

def set_cartpole_physical_params(env, pole_length=None, pole_mass=None, gear=None):
    """
    Set cartpole physical parameters as specified in config.
    """
    sys = getattr(env, 'sys', None)
    if sys is None:
        return
    if pole_length is not None:
        sys.geom_size = sys.geom_size.at[_POLE_ID, 1].set(pole_length)
    else:
        print("WARNING: Calling set_cartpole_physical_params to override them, but no value was set found in config for pole_length.")
    if pole_mass is not None:
        sys.body_mass = sys.body_mass.at[_POLE_ID].set(pole_mass)
    else:
        print("WARNING: Calling set_cartpole_physical_params to override them, but no value was set found in config for pole_mass.")
    if gear is not None:
        sys.actuator_gear = sys.actuator_gear.at[0, 0].set(gear)
    else:
        print("WARNING: Calling set_cartpole_physical_params to override them, but no value was set found in config for gear.")

class ConstraintWrapper(Wrapper):
    def __init__(self, env: MjxEnv, slider_position_bound: float):
        super().__init__(env)
        self.slider_position_bound = slider_position_bound

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        slider_pos = state.data.qpos[self.env._slider_qposadr]
        cost = (jnp.abs(slider_pos) >= self.slider_position_bound).astype(jnp.float32)
        nstate.info["cost"] = cost
        return nstate


class ActionCostWrapper(Wrapper):
    def __init__(self, env: MjxEnv, action_cost_scale: float):
        super().__init__(env)
        self.action_cost_scale = action_cost_scale

    def step(self, state: State, action: jax.Array) -> State:
        action_cost = (
            self.action_cost_scale * (1 - tolerance(action, (-0.1, 0.1), 0.1))[0]
        )
        nstate = self.env.step(state, action)
        nstate = nstate.replace(reward=nstate.reward - action_cost)
        return nstate

class PerturbedCartpoleWrapper(Wrapper):
    def __init__(self, env: MjxEnv, config: dict):
        super().__init__(env)
        self.config = config

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        set_cartpole_physical_params(
            self.env,
            pole_length=self.config.get('physical_params', {}).get('pole_length', None),
            pole_mass=self.config.get('physical_params', {}).get('pole_mass', None),
            gear=self.config.get('physical_params', {}).get('gear', None),
        )
        return state

_envs = [
    env_name
    for env_name in dm_control_suite.ALL_ENVS
    if env_name.startswith("Cartpole")
]


def make_safe(name, **kwargs):
    limit = kwargs["config"]["slider_position_bound"]
    env = dm_control_suite.load(name, **kwargs)
    env = ConstraintWrapper(env, limit)
    return env


def make_hard(name, **kwargs):
    scale = kwargs["config"]["action_cost_scale"]
    env = dm_control_suite.load(name, **kwargs)
    env = ActionCostWrapper(env, scale)
    return env

def make_safe_perturbed(name, **kwargs):
    config = kwargs.get('config', {})
    limit = config.get('slider_position_bound', None)
    env = dm_control_suite.load(name, **kwargs)
    env = ConstraintWrapper(env, limit)
    env = PerturbedCartpoleWrapper(env, config)
    return env


for env_name in _envs:
    dm_control_suite.register_environment(
        f"Safe{env_name}",
        functools.partial(make_safe, env_name),
        dm_control_suite.cartpole.default_config,
    )

for env_name in _envs:
    dm_control_suite.register_environment(
        f"Hard{env_name}",
        functools.partial(make_hard, env_name),
        dm_control_suite.cartpole.default_config,
    )

for env_name in _envs:
    dm_control_suite.register_environment(
        f"Safe{env_name}Perturbed",
        functools.partial(make_safe_perturbed, env_name),
        dm_control_suite.cartpole.default_config,
    )