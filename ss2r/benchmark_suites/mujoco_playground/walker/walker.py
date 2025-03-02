import jax
import jax.numpy as jnp
from brax.envs import Env, Wrapper
from brax.envs.base import State
from mujoco_playground import dm_control_suite

_TORSO_ID = 1
_LEFT_THIGH_ID = 5
_RIGHT_THIGH_ID = 2


def domain_randomization(sys, rng, cfg):
    @jax.vmap
    def randomize(rng):
        #  https://github.com/google-research/realworldrl_suite/blob/be7a51cffa7f5f9cb77a387c16bad209e0f851f8/realworldrl_suite/environments/walker.py#L593
        pos = sys.body_pos.copy()
        torso_length_sample = jax.random.uniform(
            rng, minval=cfg.torso_length[0], maxval=cfg.torso_length[1]
        )
        torso_length_sample = jnp.clip(torso_length_sample, a_min=-0.3, a_max=0.4)
        length = 0.3 + torso_length_sample
        scale_factor = length / 0.3
        geom = sys.geom_size.at[_TORSO_ID, 1].add(torso_length_sample)
        pos = pos.at[_TORSO_ID, -1].add(torso_length_sample)
        mass = sys.body_mass.at[_TORSO_ID].multiply(scale_factor)
        inertia = sys.body_inertia.at[_TORSO_ID].multiply(scale_factor**3)
        pos = pos.at[_LEFT_THIGH_ID, -1].add(torso_length_sample)
        pos = pos.at[_RIGHT_THIGH_ID, -1].add(torso_length_sample)
        friction_sample = jax.random.uniform(
            rng, minval=cfg.friction[0], maxval=cfg.friction[1]
        )
        friction = sys.geom_friction.at[:, 0].add(friction_sample)
        return (
            pos,
            mass,
            inertia,
            geom,
            friction,
            jnp.hstack([friction_sample, torso_length_sample]),
        )

    pos, mass, inertia, geom, friction, samples = randomize(rng)
    in_axes = jax.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace(
        {
            "body_pos": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "geom_size": 0,
            "geom_friction": 0,
        }
    )
    sys = sys.tree_replace(
        {
            "body_pos": pos,
            "body_mass": mass,
            "body_inertia": inertia,
            "geom_size": geom,
            "geom_friction": friction,
        }
    )
    return sys, in_axes, samples


class ConstraintWrapper(Wrapper):
    def __init__(self, env: Env, limit: float):
        assert isinstance(env, dm_control_suite.walker.PlanarWalker)
        super().__init__(env)
        self.limit = limit

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["cost"] = jnp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)
        joint_velocities = nstate.data.qvel[3:]
        cost = jnp.less(jnp.max(jnp.abs(joint_velocities)), self.limit).astype(
            jnp.float32
        )
        nstate.info["cost"] = cost
        return nstate


for run in [True, False]:
    name = ["Walker"]
    run_str = "Run" if run else "Walk"

    def make(**kwargs):
        angular_velocity_limit = kwargs.pop("joint_velocity_limit", 16.25)
        env = dm_control_suite.load(f"Walker{run_str}", **kwargs)
        env = ConstraintWrapper(env, angular_velocity_limit)
        return env

    name_str = f"SafeWalker{run_str}"
    dm_control_suite.register_environment(
        name_str, make, dm_control_suite.walker.default_config
    )
    dm_control_suite.ALL += [name_str]
